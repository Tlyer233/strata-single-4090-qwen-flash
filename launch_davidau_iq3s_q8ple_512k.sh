#!/bin/bash
# 4090 单卡 512K：DavidAU Qwen3.8-Flash-Next Abliterated Uncensored Turbo Brilliance 50x GSQ-RCO IQ3_S + Q8_0 PLE 表（用户指定用现成 Q8_0，不用它自带 IQ4_NL shard 2）。
# 从 3080 的 launch_swift_abliterated_iq3s_q8ple_512k.sh 移植 (2026-10-18)，差异只有三处：
#   gpus=0 单卡（没有 layer split）、路径换成 4090 的（models/Qwen-Flash-Next、venvs/cuda13）、
#   --prompt-cache-tail 官方就能开（0.1.45 help 原文 "--prompt-cache-tail  --serve, single GPU"，
#   3080 那份多卡补丁在这里不需要）。
# 512K: ctx 524288 + YaRN x2 (--rope-scaling yarn --rope-scale 2，官方 >262K 口径)。
#   依据 Strata DETAILS.md 的 IQ3_S 实测：293K NLL yarn2 2.04 vs 裸 2.22，<=32K 无退化，421K yarn2 QA 8/8。
set -euo pipefail
# 温度不钉: 模型页推荐 thinking 1.0/0.95/20, instruct 0.7/0.8/20 + presence_penalty 1.5, 与 Strata 内置
#   model card 逐字相同 (DETAILS.md:1373-1378)。SAMPLING_FORCE_TEMP 空 -> helper 删掉 sampling 块。
export SAMPLING_FORCE_TEMP="${SAMPLING_FORCE_TEMP:-}"
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
PORT="${PORT:-8005}"
CTX="${CTX:-524288}"
# --- --prefill = prompt 块大小 (token)。手写值不受 8192 的 auto 天花板管（源码 609-617 行：
#   "A chunk the user names is not capped"；两处 8192 回夹只在多卡 layer split 和 AMD gfx12 分支）。
#   原理：>=1024 token 的块会把 GPU 没装的专家整份流一遍，块越大每 token 流的字节越少（官方 #282）。
#   4090 本机实测 (0.1.45, ctx 524288, yarn2, --scenarios --runs 1；decode = 各档 10 类场景 AVG)：
#     块      32k        200k        512k       prompt 借显存 / RAM pinned 专家补品
#     8192   115.6/4438  110.4/4360   96.6/3922   3.40 GiB
#     16384  114.6/4950  106.6/4761   95.8/4217   6.41 GiB  / 20.28 GiB
#     32768  126.7/4985  112.6/4900   99.8/4335   12.43 GiB / 26.29 GiB   <- 定为默认
#     32768  116.3/5027  108.4/4896   97.3/4346   (复测第二轮：prefill +13.3/+12.3/+10.8%，decode 持平)
#     65534  123.8/4986  108.0/4869  103.0/4324   24.45 GiB / 38.32 GiB   (比 32768 还慢 0.5~0.8%)
#     65536  125.6/4998  崩           崩          24.45 GiB / 38.32 GiB
#   拐点在 32768：再往上借的显存翻倍把 ubatch 的收益吃回去，65536 更是在 200k 冷 prefill 直接打死引擎——
#     embed_rows_kernel<<<dim3(n_embd/256, n_tok)>>> 把 token 数放进 gridDim.y，CUDA 上限 65535，
#     65536 行超一格 -> launch invalid argument -> iq_embed_rows exit(1)。停放槽 4->3 不影响（不是内存问题）。
#     native_rope / native_gdn_preprocess / qsa_decode_attn 都有 >65535 的保护，只有 embed_rows 漏了。
#   回旧行为：PREFILL=8192 bash 本脚本。
PREFILL="${PREFILL:-32768}"
VRAM_RESERVE="${VRAM_RESERVE:-1024}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
KV="${KV:-k8v4}"
GPUS="${GPUS:-0}"
MTP="${MTP_DIR:-$HOME/src/Strata-data/mtp/rt}"
PLE="${PLE_GGUF:-$HOME/models/Qwen-Flash-Next/0_Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-00002-of-00002.gguf}"
ROPE_SCALING="${ROPE_SCALING:-yarn}"
ROPE_SCALE="${ROPE_SCALE:-2}"
RUN="${RUN:-$ROOT/run-davidau-iq3s-q8ple.sh}"
CFG="${CFG:-$ROOT/strata-davidau-iq3_s-q8ple.json}"
CUDA_PIP="$HOME/venvs/cuda13/lib/python3.12/site-packages/nvidia/cu13"
HAND="$HOME/ubuntu_handoff/strata"

ulimit -l unlimited 2>/dev/null || true

if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: 跳过 drain/pkill, 不碰运行中的引擎" >&2
else
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user stop sglang-dspark 2>/dev/null || true
  fi
  if [[ -x "$HAND/gpu-drain.sh" ]]; then
    bash "$HAND/gpu-drain.sh"
  fi
  # 2026-10-28: 先杀 supervisor。serve/server.py 是进程看护，只 pkill 引擎它会把引擎原地复活
  #   （今天实测：nvfp4 那份 serve 的 python 活着 -> drain 之后它重启了自己的引擎，新起的 davidau 直接 OOM）
  pkill -f "[s]erve.server" 2>/dev/null || true
  sleep 2
  pkill -x strata 2>/dev/null || true
  pkill -x strata-vision 2>/dev/null || true
  fuser -k 8000/tcp 8001/tcp 8080/tcp "${PORT}/tcp" 2>/dev/null || true
  sleep 2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2 || true
  if fuser "${PORT}/tcp" >/dev/null 2>&1; then
    echo "port $PORT still busy" >&2
    ss -ltnp | grep ":${PORT} " >&2 || true
    exit 1
  fi
fi

if [[ ! -x "$RUN" || ! -f "$CFG" ]]; then
  echo "missing $RUN / $CFG" >&2
  exit 1
fi
if [[ ! -f "$MTP/experts.bin" ]]; then
  echo "missing Swift MTP at $MTP" >&2
  exit 1
fi
if [[ ! -f "$PLE" ]]; then
  echo "missing Q8_0 PLE $PLE" >&2
  exit 1
fi
"$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" --config "$CFG" \
  --context "$CTX" --prefill "$PREFILL" --vram-reserve-mib "$VRAM_RESERVE" \
  --spec-min-p "$SPEC_MIN_P" --kv "$KV" --mtp "$MTP" --ple-gguf "$PLE" \
  --gpus "$GPUS" --resident-experts --pipeline-windows "${PIPELINE_WINDOWS:-2}"

if ! grep -q -- '--resident-experts' "$CFG"; then
  echo "refusing $CFG: IQ3_S 需要 --resident-experts (显存缓存优先, 余量进 RAM)" >&2
  exit 1
fi
if ! grep -q -- '--ple-gguf' "$CFG"; then
  echo "refusing $CFG: Q8 PLE 路径需要 --ple-gguf" >&2
  exit 1
fi
if grep -q -- '--mmap-experts' "$CFG"; then
  echo "refusing $CFG: 裸 --mmap-experts 会让 RAM 空着" >&2
  exit 1
fi

# --- 会话停放 (conversation cache): 切会话不重算 prefill (Mac 问一句、切回 dsh / 子代理) ---
# 快照只加宿主 RAM, 不占显存; 装不下就 skip parking (降级重读, 不崩)。
# 单卡没有 layer split -> 停放全支持, 磁盘层 (--conversation-cache-spill-dir) 也只有单卡能用。
# 512K 的账 (同量化实测口径 ~20-35 KB/token): 一条满长 512K 快照 ~10-15 G,
#   预算 19456 MiB 只装得下 1-2 条满长; 300K 级会话 ~6 G -> 槽 4 对 <=300K 才稳。
#   槽数超了不报错, 引擎按 LRU 驱逐 (正在打字的那条不算槽)。
# 准入公式: 快照 MiB + 地板 MiB <= MemAvailable。本机 RAM 62 G, 引擎 RSS ~24 G -> MemAvailable ~37 G,
#   停放预算 24 G / 4 槽 (与 4090 nvfp4 那份一致): iq3 专家常驻 26.29 GiB, 62 G 机器有余量。
#
# --- 旗杆(检查点) vs 停放槽, 别混 ---
# 停放 --conversation-cache-slots: 整段会话 KV 拷到内存, 换会话用。
# 旗杆 --prompt-cache: 同一段对话路上的存档, 分叉时退到最近一根再走 (~118 MB RAM/根, 不占显存)。
#   池 = N 根, 每根覆盖 --prompt-cache-every 个 token。512K 会话要 524288/131072 = 4 根 + 尾巴 1 根。
#   池 24 ≈ 2.8 G RAM, 够 4-5 段长会话互不挤占 (官方 issue #1920 的饥饿就是池太小 + every 太密)。
#   every 131072 的代价: 回退/分支最多重读 128K token; 线性 agent 轮次不受影响。
#
# --- Codex 插话 (尾部分叉) 与 --prompt-cache-tail ---
# Codex 插话 = 掐断当前请求、丢掉助手半截、把新句子接在上一句完整话后面重发 -> 尾部分叉。
# 没有尾巴旗杆时只能退到最近一根 (最坏走满 128K)。开了 tail: 每次快读完时再在尾巴插一根,
#   下一轮只重读最后几千 token (一个 --prefill chunk；PREFILL=32768 时最坏 32768 tok ~7.6 s)。
#   3080 实测 (0.1.44.1 补丁, 22141 tok): 无尾巴旗杆 0 reused / 11.4 s; 有 = 16384 reused + 5757 / 4.9 s。
#   官方 generate.cpp 把这条写死 !multi_gpu, 所以只有单卡能直接吃 -> 4090 正是单卡, 无需补丁。
# PROMPT_CACHE_TAIL=0 关掉写入; 池满时最先丢掉这根尾巴旗杆; 多占 ~118 MB RAM, 0 显存。
# --conversation-cache-min-tokens 18432 (官方 PR #1597): 短请求不再占停放槽把长会话顶掉
#   2026-10-22: 门槛从 4096 抬到 18432。主会话链根 18684（完整工具表）。
#   get_goal 那份短工具表停下来是 15036/15763，低于门槛，不占格。
#   含完整 HPP 的会话长度不低于链根，仍占格。插话本身不靠这条，靠 same-chat rewind。
#   (那条实测: 3 条几百 token 的短请求顶掉 19198 token 长会话 -> 41.4 s 重读; 设门槛后 0.3 s, 304x)。
#
# --- 2026-10-18 实测调参（Codex 插话把 448K 会话打成整段重读）---
# 现场: prompt 448411 = 0 reused + 219976 read in 47.2 s (cancelled)，重发 = 154440 reused + 294012 read in 68.4 s
#   -> 一次插话烧掉 ~115 s；不插话同一发只要 1.4 s (446025 reused + 19 read)。
# 真凶: 池 24 顶满 (日志连排 12/13/14 checkpoints, evictions=1)，而官方规则是「尾巴旗杆最先被逐」
#   (docs: "The tail checkpoint is evicted first") -> 插话退到 154440 那根，重读 294K。
# 停放实测: parked=4 bytes=15.37 GB，单条 444K 快照 snapshot_bytes=7.35 GB = 15.6 KB/token。
#   -> 停放才是 RAM 大头 (available 只剩 11 G，其中 26.29 GiB 是 PREFILL=32768 的 RAM 侧专家补品)。
# 所以: 槽 4->3 (腾 ~7 G)，池 24->40 (不让尾巴旗杆被逐；118 MB/根，写出才占，40 根上限 4.7 G)。
# 每段 512K 会话在 every=131072 下只需 4 根周期 + 1 根尾巴，池 40 = 够 8 段互不挤。
#
# --- 2026-10-18 第二轮：旗杆不是主犯，停放快照被逐才是 ---
# 逐字翻日志 (strata-swift-abliterated-iq3_s-q8ple.log 23200-23500 行)：
#   线性轮次一直是好的 —— 443219 = 443197 reused + 22 read in 174 ms。插话那两轮才炸。
#   炸的现场全是停放被逐：parked=2 evictions=4 -> 0 reused + 453158 read in 103.0 s
#                            skip parking (need 6918 MiB plus 4096 MiB floor, 7150 MiB available;
#                            evicted 1, 0 still parked) -> 0 reused + 463510 read in 104.9 s
#   引擎自己写明这一条 (generate.cpp:8972 物理 RAM 闸门): "refusing here throws away the whole
#   prompt read that produced this snapshot (measured: 270k tokens, 96 s)" —— 就是本机这句。
# 旗杆 (--prompt-cache) 只存 GDN/PLE/QSA 状态 ~118 MB/根，不存 KV；KV 全在停放快照里 (460K = 7.35 G)。
#   所以停放一被逐，旗杆再多也只能从 0 读 —— 154440 reused 那次是取消的那发刚插的根。
# RAM 账 (总 62.7 G): 引擎底 (专家补品 26.29 G + dense + CUDA) ~33.1 G，一条 460K 快照 7.35 G
#   -> 3 槽 = 33.1 + 22.1 = 55.2 G，available 只剩 7.5 G < 6.9+4.0 闸门 -> 第 3 条必被逐。
#   2 槽 = 47.8 G，available 16.4 G -> 稳。要 3 槽就得 PREFILL=16384 (补品 26.29->20.28 G，腾 6 G)。
# 两个真能降重读的开关:
#   1) --prompt-cache-spread (0.1.45 官方, help 原文): "a client that rewrites its history N tokens
#      back reads about N again, not the whole conversation" —— 插话就是改写历史, 逐字对上。
#      保留策略从 LRU 改成按距离几何摊开 (近密远疏), 池满时也不会把尾巴附近掏空。
#   2) every 131072 -> 32768: 旗杆密 4 倍, 460K 会话 14 根周期 + ~20 根轮次 + 1 尾巴 < 池 40,
#      池不溢出不驱逐; 118 MB/根 -> 上限 4.7 G RAM, 0 显存。KV 部分存活时能退得更深。
# --- 2026-10-19 回退: 尾巴旗杆 / 摊开保留 / 停放槽 全删, 只留 live 续读 ---
# 只写 --prompt-cache N + --prompt-cache-every E: 这是当前会话的 live 续读, 正常对话不重读全靠它。
#   源码 generate.cpp:14483  live_ok = o.prompt_cache > 0 && req_ckpt  ->  设成 0 就是每轮 token 0 全量 prefill。
# 停放 = 整段会话 KV 快照拷进宿主 RAM, 切会话/切 slot 靠它; 旗杆只管同一条对话的续写。
#   --conversation-cache-mib 0 等于关停放: 切走再切回必然整段 prefill (2026-10-19 踩过)。
#   min-tokens 18432: 短工具表不占槽, 不把长会话顶掉 (官方 PR #1597; 2026-10-22 从 4096 抬高)。
#   min-free-mib 2560: RAM 见底时 skip parking, 不硬挤。
# 上面 DROP_PREFIX 把 json 里所有 --conversation-cache* / --prompt-cache* 旧值一次清干净再写回。
# --- 2026-10-19 深夜: 池子要盖得住会话长度, 否则插话(改写历史)只能从 token 0 重读 ---
# 实测证据 (本机日志): prompt 213873 tokens = 0 reused + 213873 read in 45361 ms, 6 checkpoints
#   池 6 根 x every 16384 = 只覆盖 98304 token; 会话 213873 token 要 14 根 -> 分叉点在覆盖外 -> 全量重读。
# 所以 N 按 ctx/every 给足: 524288 / 16384 = 32 根。官方 help: 每根 ~118 MB RAM, 不占显存 -> 3.8 G。
#   --prompt-cache-spread (官方 help 原文): "a client that rewrites its history N tokens back reads about
#     N again, not the whole conversation" —— 就是 Codex 插话这一类, 检查点从会话尾往前摊开存。
#   --prompt-cache-tail (官方 help: --serve, single GPU; 4090 单卡符合): 在 prompt 尾部一个已有 chunk
#     边界多存一根, docs/PROMPT_CACHE_TAIL.md: "evicted first (only one stays alive, the root is never
#     dropped)", 不会挤掉周期旗杆。
# 2026-10-19 深夜二次回退: 32 根 + spread + tail 实测仍全量重读 (插话后那条请求照样 0 reused),
#   所以回到你验证过的官方 6 x 16384。想再试: PROMPT_CACHE_N=32 PROMPT_CACHE_SPREAD=1 PROMPT_CACHE_TAIL=1。
PROMPT_CACHE_N="${PROMPT_CACHE_N:-6}"
PROMPT_CACHE_EVERY="${PROMPT_CACHE_EVERY:-16384}"
PROMPT_CACHE_SPREAD="${PROMPT_CACHE_SPREAD:-0}"
PROMPT_CACHE_TAIL="${PROMPT_CACHE_TAIL:-0}"
CONV_CACHE_MIB="${CONV_CACHE_MIB:-24576}"
CONV_CACHE_SLOTS="${CONV_CACHE_SLOTS:-12}"
CONV_CACHE_MIN_TOKENS="${CONV_CACHE_MIN_TOKENS:-18432}"
CONV_CACHE_MIN_FREE_MIB="${CONV_CACHE_MIN_FREE_MIB:-2560}"
CONV_CACHE_PARK_GAIN="${CONV_CACHE_PARK_GAIN:-16384}"
CCPY_BIN="$ROOT/.venv/bin/python"
[ -x "$CCPY_BIN" ] || CCPY_BIN="$(command -v python3)"
"$CCPY_BIN" - "$CFG" "$PROMPT_CACHE_N" "$PROMPT_CACHE_EVERY" \
  "$CONV_CACHE_MIB" "$CONV_CACHE_SLOTS" "$CONV_CACHE_MIN_TOKENS" "$CONV_CACHE_MIN_FREE_MIB" \
  "$CONV_CACHE_PARK_GAIN" "$PROMPT_CACHE_SPREAD" "$PROMPT_CACHE_TAIL" <<'PYCACHE'
import json, os, shutil, sys
path, n, every, mib, slots, mintok, minfree, parkgain, spread, tail = sys.argv[1:11]
DROP_PREFIX = ("--conversation-cache", "--prompt-cache")
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
old = list(cfg.get("args") or [])
args, i = [], 0
dropped = 0
while i < len(old):
    if any(old[i].startswith(p) for p in DROP_PREFIX):
        dropped += 1
        i += 1
        if i < len(old) and not str(old[i]).startswith("-"):
            i += 1
        continue
    args.append(old[i])
    i += 1
args += ["--prompt-cache", n, "--prompt-cache-every", every,
         "--conversation-cache-mib", mib, "--conversation-cache-slots", slots,
         "--conversation-cache-min-tokens", mintok, "--conversation-cache-min-free-mib", minfree,
         "--conversation-cache-park-gain", parkgain]
if spread == "1":
    args.append("--prompt-cache-spread")
if tail == "1":
    args.append("--prompt-cache-tail")
CACHE_FLAGS = ("--prompt-cache", "--prompt-cache-every", "--conversation-cache-mib",
               "--conversation-cache-slots", "--conversation-cache-min-tokens",
               "--conversation-cache-min-free-mib", "--conversation-cache-park-gain")
if args != old:
    if not os.path.exists(path + ".bak-revert"):
        shutil.copy2(path, path + ".bak-revert")
    cfg["args"] = args
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
shown = " ".join(a + " " + args[j + 1] for j, a in enumerate(args) if a in CACHE_FLAGS)
onflags = " ".join(a for a in args if a in ("--prompt-cache-spread", "--prompt-cache-tail"))
print(f"cache: 清掉 {dropped} 个旧开关, 现在写: {shown} {onflags}", flush=True)
PYCACHE
echo "cache: --prompt-cache ${PROMPT_CACHE_N} --prompt-cache-every ${PROMPT_CACHE_EVERY} --conversation-cache-mib ${CONV_CACHE_MIB} --conversation-cache-slots ${CONV_CACHE_SLOTS} --conversation-cache-min-tokens ${CONV_CACHE_MIN_TOKENS} --conversation-cache-min-free-mib ${CONV_CACHE_MIN_FREE_MIB} park-gain=${CONV_CACHE_PARK_GAIN} spread=${PROMPT_CACHE_SPREAD} tail=${PROMPT_CACHE_TAIL}" >&2


# --- 并发流: 默认单流, 开机(systemd)/手动启动都先把 parallel 清掉 ---
# 多流走同名 dual 脚本 (export PARALLEL=2 + BATCH_MODE)。0.1.45 官方口径: 卡装不下全部专家时
#   每个 slot 从 expert cache 里拿显存, 单请求 -4~11%, N 路一起只有 1.0-1.15x (加 --batch-mtp)。
PARALLEL="${PARALLEL:-1}"
BATCH_MODE="${BATCH_MODE:-draft}"
"$CCPY_BIN" - "$CFG" "$PARALLEL" "$BATCH_MODE" <<'PYP'
import json, sys
path, par, mode = sys.argv[1], int(sys.argv[2] or 1), (sys.argv[3] or "draft").lower()
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
args = list(cfg.get("args") or [])
out, i = [], 0
while i < len(args):
    if args[i] in ("--batch-groups", "--batch-mtp") or (args[i] == "--batch" and par <= 1):
        i += 1
        while i < len(args) and not str(args[i]).startswith("-"):
            i += 1
        continue
    out.append(args[i])
    i += 1
if par > 1:
    cfg["parallel"] = par
    if mode == "draft":
        out = out + ["--batch-groups", "1", "--batch-mtp"]
    # pipe: 默认就是 pipelined 组, 不加参数
else:
    cfg.pop("parallel", None)
cfg["args"] = out
after = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if open(path, encoding="utf-8").read() != after:
    with open(path, "w", encoding="utf-8") as f:
        f.write(after)
if par > 1:
    print(f"parallel: = {par} 流 mode={mode} " + ("(--batch-groups 1 + --batch-mtp, slot 有 draft)" if mode == "draft" else "(默认 pipelined 组, slot 无 draft)"), flush=True)
else:
    print("parallel: 单流 (parallel/--batch-groups/--batch-mtp 已清除)", flush=True)
PYP

# --- 模型名统一：与 3080 一致用 qwen-flash-next（GET /v1/models 就返回这个，客户端只认它）---
# Strata 是单引擎：旧名 swift-1.5-abliterated-iq3_s-q8ple 打过来照样会跑，只是不再出现在 /v1/models 列表里。
# 写在 launcher 里 = 每次启动强制一遍，json 被 setup.py / apply_adaptive_mtp.py 重写也不会掉。
# 要换别的名字：MODEL_NAME=xxx bash 本脚本。
MODEL_NAME="${MODEL_NAME:-qwen-flash-next}"
"$CCPY_BIN" - "$CFG" "$MODEL_NAME" <<'PYNAME'
import json, sys
path, name = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
if cfg.get("model_name") != name:
    cfg["model_name"] = name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
print("model_name:", name, flush=True)
PYNAME

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$CUDA_PIP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset STRATA_ARENA_PIN_GIB
unset STRATA_RESIDENT_PIN
unset STRATA_STAGE_PIN
unset STRATA_GDN_CHUNKED
export STRATA_PREFILL_HELP="${STRATA_PREFILL_HELP:-1}"
export STRATA_PREFILL_CPU_SHARE="${STRATA_PREFILL_CPU_SHARE:-auto}"
export STRATA_PREFILL_CPU_SHARE_MAX="${STRATA_PREFILL_CPU_SHARE_MAX:-3072}"
export STRATA_STAGE_TRIM="${STRATA_STAGE_TRIM:-1}"
cd "$ROOT"
echo "Strata Swift abliterated IQ3_S + Q8_0 PLE  gpus=$GPUS(单卡)  ctx=$CTX  rope=${ROPE_SCALING}x${ROPE_SCALE}  kv=$KV  vision(gpu)  MTP=swift  PLE=sleepyeldrazi-Q8  resident-experts (GPU then RAM)  prefill=$PREFILL  reserve=${VRAM_RESERVE}  pipeline-windows=${PIPELINE_WINDOWS:-2}  min_p=$SPEC_MIN_P  parallel=$PARALLEL  PREFILL_HELP=${STRATA_PREFILL_HELP}  CPU_SHARE=${STRATA_PREFILL_CPU_SHARE}/${STRATA_PREFILL_CPU_SHARE_MAX}  STAGE_TRIM=${STRATA_STAGE_TRIM}  host=0.0.0.0  port=$PORT" >&2
if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: json 已按 parallel=${PARALLEL} 写好, 不启动引擎" >&2
  exit 0
fi
exec bash "$RUN"

#!/bin/bash
# 4090：DavidAU Qwen3.8-Flash-Next Abliterated IQ3_S + sleepyeldrazi Q8_0 PLE，512K，单卡。
# 与 launch_davidau_iq3s_q8ple_512k.sh 逐项一致（ctx/prefill/kv/PLE/resident-experts/pipeline-windows/
# conversation-cache 全套），**只把投机解码从 MTP 换成 DFlash2**（2026-11-09，引擎 v0.2.12 / 0.1.58）。
#
# 口径取自引擎自带的 strata_control/drafting.py（v0.2.12）里 dflash2 的推荐组合：
#   --draft dflash2 --draft-model <pack> --draft-fallback none --mtp-draft-vocab <file>
#   --spec 6 --draft-conf 0.35…，且**不带 --mtp**
# drafting.py 原文：dflash2 的 pack 由 tools/dflash2_pack.py --q4_0 从 incoai/Qwen3.8-Flash-Next-DFlash2
#   的 1.2 GB safetensors 在本机生成（522,458,112 B）；“A Q8 pack (760,353,792 bytes) made by an earlier
#   Strata keeps working and is never fetched again; Q4_0 and Q8 ran the same speed (-0.3%)”。
#   本机盘上正好有这份 Q8 pack（760,353,792 B），4090 又没有外网，所以默认用它；要换 Q4_0/调过的 pack，
#   只需 DF2_PACK=... 覆盖（脚本会按两种合法尺寸校验）。
#
# 注意：--draft-fallback none = 只加载 DFlash2，起不来直接报错（省 ~0.9 GiB 显存）。
# 用法（本脚本不自动启动；你确认 GPU 空闲后自己跑）：
#   bash launch_davidau_iq3s_q8ple_dflash2_512k.sh
#   JSON_ONLY=1 bash ...            # 只写 json，不碰引擎
#   DF2_PACK=/path/to/x.pack bash ...
set -euo pipefail
ROOT="${STRATA_ROOT:-$HOME/src/Strata}"
PORT="${PORT:-8005}"
CTX="${CTX:-524288}"
PREFILL="${PREFILL:-32768}"
VRAM_RESERVE="${VRAM_RESERVE:-1024}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
KV="${KV:-k8v4}"
GPUS="${GPUS:-0}"
MTP="${MTP_DIR:-$HOME/src/Strata-data/mtp/rt}"
PLE="${PLE_GGUF:-$HOME/models/Qwen-Flash-Next/0_Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-00002-of-00002.gguf}"
DF2_PACK="${DF2_PACK:-$HOME/src/Strata-data/dflash2/qwen3.8-flash-next-dflash2.q8.pack}"
DF2_CONF="${DF2_CONF:-0.35}"
DF2_SPEC="${DF2_SPEC:-6}"
ROPE_SCALING="${ROPE_SCALING:-yarn}"
ROPE_SCALE="${ROPE_SCALE:-2}"
BASE_CFG="${BASE_CFG:-$ROOT/strata-davidau-iq3_s-q8ple.json}"
export STRATA_UNDECLARED_TOOL_CALLS="${STRATA_UNDECLARED_TOOL_CALLS:-call}"  # 4622: undeclared-but-well-formed calls return as calls
CFG="${CFG:-$ROOT/strata-davidau-iq3_s-q8ple-dflash2.json}"
RUN="${RUN:-$ROOT/run-davidau-iq3s-q8ple.sh}"
HAND="$HOME/ubuntu_handoff/strata"

ulimit -l unlimited 2>/dev/null || true

# ---------- DFlash2 pack 校验（两种合法尺寸） ----------
if [[ ! -f "$DF2_PACK" ]]; then
  echo "missing DFlash2 pack: $DF2_PACK" >&2
  echo "  合法尺寸: Q4_0 522,458,112 B（tools/dflash2_pack.py --q4_0）或 Q8 760,353,792 B" >&2
  exit 1
fi
PACK_SIZE=$(stat -c%s "$DF2_PACK")
case "$PACK_SIZE" in
  522458112|760353792) echo "df2 pack: $DF2_PACK ($PACK_SIZE B)" >&2 ;;
  *) echo "refusing $DF2_PACK: size $PACK_SIZE 不是 Strata 已知的 Q4_0/Q8 pack" >&2; exit 1 ;;
esac
if ! "$ROOT/engine/strata" --help 2>&1 | grep -q -- '--draft dflash2'; then
  echo "refusing: $ROOT/engine/strata 没有 DFlash2（需要 0.1.58+）" >&2
  exit 1
fi
if [[ ! -f "$MTP/draft_vocab.bin" ]]; then
  echo "missing $MTP/draft_vocab.bin（--mtp-draft-vocab 要它）" >&2
  exit 1
fi

# ---------- 停/清 其它引擎（与 512k 脚本同款） ----------
if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: 跳过 drain/pkill，不碰运行中的引擎" >&2
else
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user stop sglang-dspark 2>/dev/null || true
  fi
  if [[ -x "$HAND/gpu-drain.sh" ]]; then
    bash "$HAND/gpu-drain.sh"
  fi
  pkill -f "[s]erve.server" 2>/dev/null || true
  sleep 2
  pkill -x strata 2>/dev/null || true
  pkill -x strata-vision 2>/dev/null || true
  fuser -k 8000/tcp 8001/tcp 8080/tcp "${MODE_PROXY_PORT:-8010}/tcp" "${PORT}/tcp" 2>/dev/null || true
  sleep 2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2 || true
  if fuser "${PORT}/tcp" >/dev/null 2>&1; then
    echo "port $PORT still busy" >&2
    ss -ltnp | grep ":${PORT} " >&2 || true
    exit 1
  fi
fi

if [[ ! -x "$RUN" ]]; then echo "missing $RUN" >&2; exit 1; fi
if [[ ! -f "$PLE" ]]; then echo "missing Q8_0 PLE $PLE" >&2; exit 1; fi

# ---------- 独立配置：从 512k 的基础 config 复制一份，别让两条脚本互相改 ----------
if [[ ! -f "$CFG" ]]; then
  [[ -f "$BASE_CFG" ]] || { echo "missing base config $BASE_CFG" >&2; exit 1; }
  cp -a "$BASE_CFG" "$CFG"
  [[ -f "$BASE_CFG.bak-revert" && ! -f "$CFG.bak-revert" ]] && cp -a "$BASE_CFG.bak-revert" "$CFG.bak-revert" || true
  echo "seeded $CFG from $BASE_CFG" >&2
fi

# ---------- 模型/pack/显存 等基础参数（与 512k 完全相同） ----------
"$ROOT/.venv/bin/python" -u "$HAND/apply_adaptive_mtp.py" --config "$CFG" \
  --context "$CTX" --prefill "$PREFILL" --vram-reserve-mib "$VRAM_RESERVE" \
  --spec-min-p "$SPEC_MIN_P" --kv "$KV" --mtp "$MTP" --ple-gguf "$PLE" \
  --gpus "$GPUS" --resident-experts --pipeline-windows "${PIPELINE_WINDOWS:-2}"

for need in --resident-experts --ple-gguf; do
  grep -q -- "$need" "$CFG" || { echo "refusing $CFG: 缺 $need" >&2; exit 1; }
done
if grep -q -- '--mmap-experts' "$CFG"; then
  echo "refusing $CFG: 裸 --mmap-experts 会让 RAM 空着" >&2
  exit 1
fi

# ---------- 把 MTP 投机解码换成 DFlash2（只改 spec 相关项） ----------
"$ROOT/.venv/bin/python" - "$CFG" "$DF2_PACK" "$MTP/draft_vocab.bin" "$DF2_SPEC" "$DF2_CONF" <<'PYDF2'
import json, os, shutil, sys
path, pack, vocab, spec, conf = sys.argv[1:6]
# 这些是 MTP 专属或会被 DFlash2 取代的项：整对删除
DROP_PAIRS = ("--mtp", "--mtp-max-t", "--suffix-draft", "--mtp-hnorm", "--mtp-q4",
              "--mtp-draft-vocab", "--draft", "--draft-model", "--draft-fallback",
              "--draft-conf", "--draft-cap", "--lookup-chain")
# 纯开关（无值）也要清掉
DROP_FLAGS = ("--coupled-draft",)
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
old = list(cfg.get("args") or [])
args, i, dropped = [], 0, []
while i < len(old):
    a = old[i]
    if a in DROP_PAIRS:
        dropped.append(a)
        i += 1
        if i < len(old) and not str(old[i]).startswith("--"):
            i += 1
        continue
    if a in DROP_FLAGS:
        dropped.append(a); i += 1; continue
    args.append(a); i += 1

def upsert(args, flag, value):
    out, i, seen = [], 0, False
    while i < len(args):
        if args[i] == flag:
            seen = True
            i += 1
            if i < len(args) and not str(args[i]).startswith("--"):
                i += 1
            continue
        out.append(args[i]); i += 1
    out += [flag, value]
    return out

args = upsert(args, "--spec", spec)
args += ["--draft", "dflash2",
         "--draft-model", pack,
         "--draft-fallback", "none",
         "--mtp-draft-vocab", vocab,
         "--draft-conf", conf]

if args != old:
    if not os.path.exists(path + ".bak-dflash2"):
        shutil.copy2(path, path + ".bak-dflash2")
    cfg["args"] = args
    cfg["drafting"] = "dflash2"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
print("df2: 清掉 " + (" ".join(dropped) if dropped else "(无)"), flush=True)
print("df2: --spec %s --draft dflash2 --draft-model %s --draft-fallback none --draft-conf %s"
      % (spec, pack, conf), flush=True)
PYDF2

# ---------- 会话停放（与 512k 脚本逐字相同的块） ----------
PROMPT_CACHE_N="${PROMPT_CACHE_N:-6}"
PROMPT_CACHE_EVERY="${PROMPT_CACHE_EVERY:-16384}"
PROMPT_CACHE_SPREAD="${PROMPT_CACHE_SPREAD:-0}"
PROMPT_CACHE_TAIL="${PROMPT_CACHE_TAIL:-0}"
CONV_CACHE_MIB="${CONV_CACHE_MIB:-24576}"
CONV_CACHE_SLOTS="${CONV_CACHE_SLOTS:-24}"
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

echo "Strata DavidAU IQ3_S + Q8_0 PLE  ctx=$CTX  rope=${ROPE_SCALING}x${ROPE_SCALE}  kv=$KV  gpus=$GPUS(单卡)  prefill=$PREFILL  reserve=$VRAM_RESERVE  pipeline-windows=${PIPELINE_WINDOWS:-2}" >&2
echo "draft=DFlash2  pack=$DF2_PACK ($PACK_SIZE B)  fallback=none  spec=$DF2_SPEC conf=$DF2_CONF  draft_vocab=$MTP/draft_vocab.bin" >&2
echo "cfg=$CFG  runner=$RUN  port=$PORT" >&2

if [ "${JSON_ONLY:-0}" = 1 ]; then
  echo "JSON_ONLY=1: json 已写好，不启动引擎" >&2
  exit 0
fi

exec bash "$RUN"

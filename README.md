# strata single-4090 · Qwen3.8-Flash-Next

* 模型: https://huggingface.co/DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF
* ngram: https://huggingface.co/sleepyeldrazi/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-GGUF/tree/main
* script: [launch_davidau_iq3s_q8ple_512k.sh](launch_davidau_iq3s_q8ple_512k.sh) · [apply_adaptive_mtp.py](apply_adaptive_mtp.py) · [scripts/decode_ctx_tps.py](scripts/decode_ctx_tps.py)

单卡 4090 48G · strata 官方 tag `v0.2.2`（`strata --version` = 0.1.47）· ctx 524288 + `--rope-scaling yarn --rope-scale 2` ·
`--kv k8v4` `--prefill 32768` `--resident-experts` `--pipeline-windows 2` 单流 · 旗杆 `6×16384` · 停放 `24576/12/18432/2560` ·
bench tg 256 · runs 1 · thinking off · `when 2026-10-25T16:05:13` · GPU 解锁后

```text
  tier    code   human    math   multi      qa     rag  reason    role    stem   write     AVG  |     pre_s  pre_min   pre_tps
------------------------------------------------------------------------------------------------------------------------------
    8k   134.5   144.6   140.1   118.8   124.9   149.6   150.7   141.1   121.6   128.7   135.5  |       1.8     0.03      4383
   32k   120.1   117.6   145.7   124.6   128.1   149.9   141.4   139.2   130.0   121.5   131.8  |       6.2     0.10      5090
   64k   112.1   137.8   140.5   130.1   136.1   128.3   141.2   137.2   134.7   126.5   132.4  |      12.1     0.20      5190
  128k   119.8   118.8   123.3   118.8   111.2   141.7   129.4   124.2   134.3   114.6   123.6  |      24.9     0.41      5026
  200k    99.6   120.2   127.4   107.4   109.5   133.7   120.2   129.7   123.0   112.3   118.3  |      39.7     0.66      4920
  256k   111.8   130.1   132.5   112.5   117.5   126.7   129.4   132.5   130.1   122.1   124.5  |      52.7     0.88      4743
  384k   120.8   110.5   126.9   112.8   114.3   127.7   134.9   115.5   115.5   115.8   119.5  |      82.1     1.37      4570
  512k   108.2    97.6   120.4   107.0   103.8   127.8   116.8   117.1   122.3   112.1   113.3  |     113.5     1.89      4402
------------------------------------------------------------------------------------------------------------------------------
   AVG   115.9   122.2   132.1   116.5   118.2   135.7   133.0   129.6   126.4   119.2   124.9
------------------------------------------------------------------------------------------------------------------------------
```

解锁前同口径 decode AVG **113.8**（512K 档 108.5 / 整段冷读 118.0 s）→ 解锁 **+9.8%**。
原始数据 [results/iq3-modeon.json](results/iq3-modeon.json)（解锁后）· [results/iq3-modeoff.json](results/iq3-modeoff.json)（解锁前）；
重打这张表：`python3 scripts/render_result.py results/iq3-modeon.json`。

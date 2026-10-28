#!/usr/bin/env python3
"""把 decode_ctx_tps.py 存的 result json 重新打成它原来那张 ASCII 表（不重跑）。
用法: render_result.py <result.json | -> [tag]        tag 只用来在表头下面标一行出处
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from decode_ctx_tps import ABBREV, scenario_table  # noqa: E402


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    raw = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()
    d = json.loads(raw)
    cats = d.get("scenarios") or [c for c in (d.get("tiers") or [{}])[0].get("scenarios", {})]
    rows = d.get("tiers") or []
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    print(scenario_table(rows, cats))
    print(f"when={d.get('when')}  engine={d.get('engine')}  n_ctx={d.get('n_ctx')}  "
          f"tg={d.get('tg')}  runs={d.get('runs')}  thinking={json.dumps(d.get('extra_body'))}  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

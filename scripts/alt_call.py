"""Send a Markdown pack to a non-Anthropic model through OpenRouter (owner tooling).

The model comes from an llm-profiles task profile (`llmp resolve <profile>`), never a
literal id. Only aggregates and code structure go out: no row-level data, nothing
personal (profiles are data class `internal`). Every call is appended to
reports/reviews/openrouter_spend.tsv; a call is refused once the ledger reaches the budget.

    .venv/bin/python -I scripts/alt_call.py PROFILE PACK.md OUT.md [--budget 10]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "reviews" / "openrouter_spend.tsv"
KEY_FILE = Path(os.environ.get("IAA_KEY_FILE", ROOT / ".env"))  # gitignored, chmod 600


def api_key(env: str) -> str:
    if os.environ.get(env):
        return os.environ[env]
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith(env + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{env} not set and not in {KEY_FILE}")


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    rows = LEDGER.read_text().splitlines()[1:]
    return sum(float(r.split("\t")[4]) for r in rows if r.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("pack", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--max-tokens", type=int, default=40000)
    a = ap.parse_args()
    if spent() >= a.budget:
        raise SystemExit(f"budget reached: ${spent():.2f} of ${a.budget:.2f}")
    prof = json.loads(
        subprocess.run(
            ["llmp", "resolve", a.profile], capture_output=True, text=True, check=True, timeout=30
        ).stdout
    )
    body = {
        "model": prof["model"],
        "messages": [{"role": "user", "content": a.pack.read_text()}],
        "reasoning": {"effort": prof["effort"]},
        "max_tokens": a.max_tokens,
        "provider": {"data_collection": "deny"},
        "usage": {"include": True},
    }
    req = urllib.request.Request(
        prof["base_url"] + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key(prof['api_key_env'])}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:  # noqa: S310 (fixed https base_url)
        out = json.load(r)
    msg = out["choices"][0]["message"].get("content") or ""
    u = out.get("usage", {})
    cost = float(u.get("cost") or 0.0)
    a.out.write_text(msg)
    if not msg.strip():  # keep the raw response to see why (finish reason, error, reasoning)
        a.out.with_suffix(".raw.json").write_text(json.dumps(out, indent=1)[:200000])
    if not LEDGER.exists():
        LEDGER.write_text(
            "timestamp\tprofile\tmodel\tpack\tcost_usd\tprompt_tokens\tcompletion_tokens\n"
        )
    with LEDGER.open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{a.profile}\t{prof['model']}\t{a.pack.name}\t"
            f"{cost:.4f}\t{u.get('prompt_tokens', 0)}\t{u.get('completion_tokens', 0)}\n"
        )
    print(
        f"{a.out.name}: {time.time() - t0:.0f}s, ${cost:.3f}, total ${spent():.2f}, {len(msg)} chars"
    )


if __name__ == "__main__":
    main()

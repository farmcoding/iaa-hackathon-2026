"""Generate concept illustrations for the dashboard via the llm-profiles
`image-diagram` profile (OpenRouter, data class public). Prompts are generic concept
descriptions; no competition data is sent. Outputs are cached; existing files are kept."""

import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
prof = json.loads(
    subprocess.run(
        ["llmp", "resolve", "image-diagram"], capture_output=True, text=True, check=True, timeout=30
    ).stdout
)
KEY = os.environ[prof["api_key_env"]]
STYLE = (
    "Clean flat corporate infographic, white background, muted blue, grey and one orange "
    "accent, simple icons, generous whitespace, legible short English labels, no logos, "
    "no people's faces, landscape 16:9."
)
PROMPTS = {
    "loop": "A circular workflow diagram titled 'Supervised experiment loop'. Five steps around "
    "a circle: 1 'AI agent proposes one change', 2 'Smoke test', 3 'Frozen scoring "
    "harness', 4 'Results ledger', 5 'Keep or revert (rule)'. Outside the circle a "
    "separate gate labelled 'Actuary sign-off' leading to 'Leaderboard submission'. "
    "A lock icon on the harness.",
    "ultimate": "An actuarial concept diagram titled 'From reported claims to ultimate cost'. "
    "A horizontal stacked bar for one accident year: 'Paid', 'Case reserves', "
    "'Late-reported claims (IBNR)', 'Reserve shortfall (IBNER)'. Above it an arrow "
    "labelled 'Development'. Small clock icons showing reporting delay. Do not show any numbers, percentages or currency amounts anywhere.",
    "folds": "A timeline diagram titled 'Out-of-time validation'. Years 2021, 2022, 2023, 2024, "
    "2025 left to right. Row 'Fold A': train 2021-2022, test 2023. Row 'Fold B': train "
    "2021-2023, test 2024. Row 'Final': train 2021-2024, predict 2025. Training blocks "
    "blue, test blocks orange, prediction block outlined.",
    "governance": "A layered pyramid diagram titled 'Governance layers'. Bottom layer 'Data "
    "checks and tests', next 'Frozen harness and ledger', next 'Human review "
    "and sign-off', top 'Professional standards (CIA, IAA)'. Shield icon.",
}


def gen(name: str, prompt: str) -> None:
    path = OUT / f"{name}.png"
    if path.exists():
        print("cached", path)
        return
    body = {
        "model": prof["model"],
        "modalities": ["image", "text"],
        "messages": [{"role": "user", "content": f"{prompt} {STYLE}"}],
    }
    req = urllib.request.Request(
        f"{prof['base_url']}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        msg = json.load(r)["choices"][0]["message"]
    url = msg["images"][0]["image_url"]["url"]
    path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
    print("wrote", path)


for n, p in PROMPTS.items():
    try:
        gen(n, p)
    except Exception as e:  # report and continue with the other images
        print("failed", n, type(e).__name__, str(e)[:200])

"""Build the experiment dashboard: one self-contained HTML file for a senior actuary.

Reads the ledger (results.tsv), saved validation predictions (data/cache/preds), the
backlog, the questions register, the data-quality log, the sign-off log and cached EDA
aggregates. Charts are Plotly (hover on every mark); concept illustrations come from
the llm-profiles `image-diagram` profile and are cached in reports/dashboard/assets.

Usage: python -I scripts/dashboard.py [--root REPO] [--out FILE] [--batch N]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path

import markdown
import yaml
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# Reference palette (dataviz skill, light mode, validated)
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
LINE_COLOR = {"motor": S[0], "buildings": S[1], "contents": S[2]}
VERDICT = {
    "keep": (S[0], "Kept by rule"),
    "keep_owner": (S[6], "Kept by owner decision"),
    "discard": (MUTED, "Discarded"),
    "crash": (CRIT, "Crashed"),
    "over_budget": (WARN, "Over time budget"),
    "unconfirmed": (WARN, "Passed once, not confirmed on fresh seeds"),
    "blend_pass": (S[2], "Blend passes keep rule"),
    "blend_fail": (MUTED, "Blend fails keep rule"),
}


def layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"},
        height=height,
        paper_bgcolor=SURF,
        plot_bgcolor=SURF,
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": INK2,
            "size": 12,
        },
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
        legend={"orientation": "h", "y": -0.18, "x": 0},
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont={"color": MUTED})
    return fig


def chart(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def img(path: Path, alt: str) -> str:
    if not path.exists():
        return ""
    return f'<img class="illus" alt="{html.escape(alt)}" src="assets/{path.name}">'


def md(text: str) -> str:
    return markdown.markdown(text, extensions=["tables"])


NOISE_FLOOR = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "market_dev.yaml").read_text())["harness"]["noise_floor"]  # today's primary margin
FOLD_FLOOR = 0.002


def why(r: pd.Series, best: float, best_min: float, delta: float) -> str:
    """Plain-English reason for the verdict, re-derived with today's keep rule."""
    v = r["verdict"]
    if v == "keep":
        conf = pd.to_numeric(r.get("confirm_primary"), errors="coerce")
        tail = f"; fresh-seed rerun {conf:.4f}" if pd.notna(conf) else ""
        return (
            f"Beat the best by {delta:+.4f} without hurting either year{tail}"
            if pd.notna(delta)
            else "First kept run"
        )
    if v == "keep_owner":
        return "Owner decision"
    if v == "unconfirmed":
        conf = pd.to_numeric(r.get("confirm_primary"), errors="coerce")
        return f"Passed once, but the fresh-seed rerun scored {conf:.4f}, not enough to pass again"
    if v == "over_budget":
        return f"Ran {r['seconds']:.0f} s against the 1,800 s limit"
    if v == "crash":
        return "Code failed before scoring (error at the start of the hypothesis text)"
    if v in ("blend_pass", "blend_fail"):
        return "Owner blend, scored outside the loop"
    if v == "discard" and pd.notna(delta):
        if delta <= 0:
            return f"Scored below the best ({delta:+.4f})"
        if delta <= NOISE_FLOOR:
            return f"Gain {delta:+.4f} is inside the {NOISE_FLOOR} seed-noise floor"
        if pd.notna(best_min) and r["min_fold"] < best_min - FOLD_FLOOR:
            return f"Worse year fell to {r['min_fold']:.4f} (best run's worse year {best_min:.4f})"
        return "Clears today's bar; it failed the bar in force then (the retired 0.1339 best, or the old equal-weight rule)"
    return ""


def load_ledger(root: Path) -> pd.DataFrame:
    rows = list(csv.DictReader((root / "results.tsv").open(), delimiter="\t"))
    df = pd.DataFrame(rows)
    for c in [
        "primary",
        "d2_A",
        "d2_B",
        "min_fold",
        "d2_motor_B",
        "d2_buildings_B",
        "d2_contents_B",
        "ae_motor_B",
        "ae_buildings_B",
        "ae_contents_B",
        "gini_B",
        "seconds",
    ]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n"] = range(1, len(df) + 1)
    # primary on the current definition (fold B weighted twice)
    df["score"] = (df["d2_A"] + 2 * df["d2_B"]) / 3
    # owner-retired keeps (config harness.superseded) never set the bar
    cfg = yaml.safe_load((root / "configs" / "market_dev.yaml").read_text())
    superseded = set(cfg["harness"].get("superseded") or [])
    best, best_min, deltas, reasons = np.nan, np.nan, [], []
    for _, r in df.iterrows():
        deltas.append(r["score"] - best if np.isfinite(best) and pd.notna(r["score"]) else np.nan)
        reasons.append(why(r, best, best_min, deltas[-1]))
        live = pd.notna(r["score"]) and r["timestamp"] not in superseded
        if live and r["verdict"] == "keep_owner":
            best, best_min = r["score"], r["min_fold"]  # an owner keep re-bases the bar
        elif live and r["verdict"] == "keep" and (not np.isfinite(best) or r["score"] > best):
            best, best_min = r["score"], r["min_fold"]
    df["delta"] = deltas
    df["why"] = reasons
    df["family"] = df["description"].fillna("").map(family)
    return df


FAMILIES = [  # checked in order: most specific first
    ("Neural blend (TabM)", r"tabm"),
    ("Reproducibility", r"re-baseline|reproduc"),
    ("Large losses", r"\bcap\b|capp|large loss|excess reload"),
    (
        "Target structure",
        r"frequency model|frequency x|frequency ×|poisson frequency|peril model|float|weather-peril|split motor",
    ),
    ("Constraints", r"monoton"),
    ("Level and calibration", r"anchor|isotonic|calibrat|true-up|balance"),
    ("Loss function", r"tweedie (variance )?power|training power"),
    (
        "Features and encoding",
        r"feature|encod|interaction|premium|weather|categor|sample_weight|weight",
    ),
    ("Ensembling", r"blend|bag|seed|ensemble|cann|glm"),
    ("Hyperparameters", r"regulari|leaves|min_data|learning rate|hyperparam"),
]


def family(desc: str) -> str:
    d = desc.lower()
    if d.startswith("baseline"):
        return "Baseline"
    for name, pat in FAMILIES:
        if re.search(pat, d):
            return name
    return "Other"


def lift_table(pred_path: Path, fold: str = "B", bins: int = 10) -> pd.DataFrame | None:
    if not pred_path.exists():
        return None
    p = pd.read_parquet(pred_path)
    p = p[p["fold"] == fold].copy()
    p["rate"] = p["pred"] / p["exposure"]
    p = p.sort_values("rate")
    cum = p["exposure"].cumsum() / p["exposure"].sum()
    p["decile"] = np.minimum((cum * bins).astype(int), bins - 1) + 1
    lc = p["y"].sum() / p["exposure"].sum()
    g = p.groupby("decile").agg(e=("exposure", "sum"), y=("y", "sum"), pr=("pred", "sum"))
    g["actual"] = g["y"] / g["e"] / lc
    g["predicted"] = g["pr"] / g["e"] / lc
    return g.reset_index()


# --------------------------------------------------------------------------- charts


def fig_progress(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    kept = df[df["verdict"].isin(["keep", "keep_owner"])]
    fig.add_trace(
        go.Scatter(
            x=kept["n"],
            y=kept["score"].cummax(),
            mode="lines",
            line={"color": S[0], "width": 2, "shape": "hv"},
            name="Best so far",
            hoverinfo="skip",
        )
    )
    for v, (color, label) in VERDICT.items():
        d = df[df["verdict"] == v]
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["n"],
                y=d["score"],
                mode="markers",
                name=label,
                marker={"size": 11, "color": color, "line": {"color": SURF, "width": 2}},
                customdata=np.stack(
                    [d["description"].str.slice(0, 120), d["d2_A"], d["d2_B"]], axis=1
                ),
                hovertemplate="Run %{x}<br>Score %{y:.4f}<br>Fold A %{customdata[1]:.4f} · Fold B %{customdata[2]:.4f}<br>%{customdata[0]}<extra></extra>",
            )
        )
    layout(fig, "Validation score, run by run (higher is better)")
    fig.update_xaxes(title="Experiment run", dtick=1)
    fig.update_yaxes(title="Score (Tweedie D², fold B weighted twice)", tickformat=".3f")
    return fig


def fig_folds(df: pd.DataFrame) -> go.Figure:
    d = df[df["verdict"].isin(["keep", "keep_owner"])]
    fig = go.Figure()
    for col, name, color in (
        ("d2_A", "Fold A (score 2023)", S[1]),
        ("d2_B", "Fold B (score 2024)", S[0]),
    ):
        fig.add_trace(
            go.Scatter(
                x=d["n"],
                y=d[col],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"size": 9},
                hovertemplate=f"{name}<br>Run %{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    layout(fig, "Kept runs: both validation years", height=340)
    fig.update_xaxes(title="Experiment run", dtick=1)
    fig.update_yaxes(title="Tweedie D²", tickformat=".3f")
    return fig


def fig_lines(df: pd.DataFrame) -> go.Figure:
    d = df[df["verdict"].isin(["keep", "keep_owner"])]
    fig = go.Figure()
    for ln in ("motor", "buildings", "contents"):
        fig.add_trace(
            go.Bar(
                x=[f"Run {n}" for n in d["n"]],
                y=d[f"d2_{ln}_B"],
                name=ln.title(),
                marker={"color": LINE_COLOR[ln], "line": {"color": SURF, "width": 2}},
                hovertemplate=f"{ln.title()}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    layout(fig, "Within-line accuracy, fold B (kept runs)", height=340, barmode="group")
    fig.update_yaxes(title="Tweedie D² within the line", tickformat=".3f")
    return fig


def fig_ae(df: pd.DataFrame) -> go.Figure:
    d = df[df["verdict"].isin(["keep", "keep_owner"])].tail(1)
    fig = go.Figure()
    if not d.empty:
        r = d.iloc[0]
        vals = [r[f"ae_{ln}_B"] for ln in ("motor", "buildings", "contents")]
        fig.add_hrect(y0=0.97, y1=1.03, fillcolor=GRID, opacity=0.6, line_width=0, layer="below")
        fig.add_hline(y=1.0, line={"color": INK2, "width": 1, "dash": "dot"}, layer="below")
        fig.add_trace(
            go.Scatter(
                x=["Motor", "Buildings", "Contents"],
                y=vals,
                mode="markers+text",
                marker={
                    "size": 16,
                    "color": [LINE_COLOR[k] for k in LINE_COLOR],
                    "line": {"color": SURF, "width": 2},
                },
                text=[f"{v:.3f}" for v in vals],
                textposition="middle right",
                hovertemplate="%{x}: actual / expected %{y:.3f}<extra></extra>",
            )
        )
    layout(
        fig,
        "Actual vs expected by line, fold B (latest kept run; grey band 0.97 to 1.03)",
        height=320,
        showlegend=False,
    )
    fig.update_yaxes(range=[0.9, 1.1], title="Actual ÷ predicted")
    fig.update_xaxes(range=[-0.5, 2.7])
    return fig


def fig_lift(df: pd.DataFrame, preds: Path) -> tuple[go.Figure | None, go.Figure | None]:
    # Prediction files are named by run start time; match each ledger row to the latest
    # file that started before the row was written (and after the previous row).
    files = sorted(preds.glob("*.parquet")) if preds.exists() else []
    starts = [datetime.strptime(f.stem, "%Y%m%dT%H%M%S") for f in files]
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    match = {}
    for i, t_end in enumerate(ts):
        t_prev = ts.iloc[i - 1] if i else pd.Timestamp.min
        cands = [f for f, s0 in zip(files, starts) if t_prev <= s0 <= t_end]
        if cands:
            match[i] = cands[-1]
    kept = df[df["verdict"].isin(["keep", "keep_owner"])]
    curves = []
    for idx, r in kept.iterrows():
        if idx not in match:
            continue
        t = lift_table(match[idx])
        if t is not None:
            curves.append((r["n"], t))
    if not curves:
        return None, None
    fig = go.Figure()
    for i, (n, t) in enumerate(curves[-4:]):
        fig.add_trace(
            go.Scatter(
                x=t["decile"],
                y=t["actual"],
                mode="lines+markers",
                name=f"Run {n}",
                line={"color": S[i], "width": 2},
                marker={"size": 8},
                hovertemplate=f"Run {n}<br>Decile %{{x}}: actual %{{y:.2f}} × average<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line={"color": INK2, "width": 1, "dash": "dot"})
    layout(fig, "Lift: actual loss cost by predicted-risk decile, fold B", height=380)
    fig.update_xaxes(title="Decile of predicted risk (1 = lowest, equal exposure)", dtick=1)
    fig.update_yaxes(title="Actual loss cost ÷ portfolio average")
    spread = go.Figure(
        go.Bar(
            x=[f"Run {n}" for n, _ in curves],
            y=[t["actual"].iloc[-1] / max(t["actual"].iloc[0], 1e-9) for _, t in curves],
            marker={"color": S[0]},
            hovertemplate="%{x}: top decile costs %{y:.1f}× the bottom decile<extra></extra>",
        )
    )
    layout(
        spread,
        "Separation: top-decile ÷ bottom-decile actual loss cost",
        height=300,
        showlegend=False,
    )
    spread.update_yaxes(title="Ratio")
    return fig, spread


def fig_family(df: pd.DataFrame) -> go.Figure:
    d = df[df["family"] != "Baseline"].dropna(subset=["delta"])
    g = (
        d.groupby("family")
        .agg(
            tried=("n", "size"),
            kept=("verdict", lambda v: v.isin(["keep", "keep_owner"]).sum()),
            best=("delta", "max"),
        )
        .reset_index()
        .sort_values("best")
    )
    fig = go.Figure(
        go.Bar(
            y=g["family"],
            x=g["best"],
            orientation="h",
            marker={"color": [S[0] if b > 0 else MUTED for b in g["best"]]},
            customdata=np.stack([g["tried"], g["kept"]], axis=1),
            hovertemplate="%{y}<br>Best change in score %{x:+.4f}<br>Runs tried %{customdata[0]}, kept %{customdata[1]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": INK2, "width": 1})
    layout(
        fig,
        "Approaches compared: best score change each produced",
        height=60 + 40 * max(len(g), 3),
        showlegend=False,
    )
    fig.update_xaxes(title="Change in score vs the best model before that run", tickformat="+.3f")
    return fig


def fig_cv_lb(root: Path) -> go.Figure | None:
    txt = (root / "docs" / "SIGNOFF.md").read_text()
    pts = []
    for line in txt.splitlines():
        m = re.search(r"\| [\d.]+ \(([\d.]+) / ([\d.]+)\) \| ([\d.]+) \|", line)
        if m:
            pts.append((float(m.group(2)), float(m.group(3)), line.split("|")[1].strip()))
    if len(pts) < 2:
        return None
    fig = go.Figure(
        go.Scatter(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            mode="markers+text",
            text=[f"Submission {i + 2}" for i in range(len(pts))],
            textposition="bottom right",
            marker={"size": 12, "color": S[0], "line": {"color": SURF, "width": 2}},
            hovertemplate="Fold B %{x:.4f}<br>Public leaderboard %{y:.4f}<extra></extra>",
        )
    )
    layout(fig, "Does local validation predict the leaderboard?", height=340, showlegend=False)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fig.update_xaxes(
        title="Local score, fold B", tickformat=".3f", range=[min(xs) - 0.003, max(xs) + 0.004]
    )
    fig.update_yaxes(
        title="Public leaderboard", tickformat=".3f", range=[min(ys) - 0.004, max(ys) + 0.003]
    )
    return fig


def fig_losscost(eda: dict) -> go.Figure:
    ly = pd.DataFrame(eda["line_year"])
    fig = go.Figure()
    for ln in ("motor", "buildings", "contents"):
        d = ly[ly["line"] == ln]
        fig.add_trace(
            go.Scatter(
                x=d["year"],
                y=d["reported"] / d["exposure"],
                mode="lines+markers",
                name=f"{ln.title()} reported",
                line={"color": LINE_COLOR[ln], "width": 2, "dash": "dot"},
                marker={"size": 8},
                hovertemplate=f"{ln.title()} reported %{{x}}: %{{y:.0f}} CU<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d["year"],
                y=d["ultimate"] / d["exposure"],
                mode="lines+markers",
                name=f"{ln.title()} developed",
                line={"color": LINE_COLOR[ln], "width": 2},
                marker={"size": 9},
                hovertemplate=f"{ln.title()} developed %{{x}}: %{{y:.0f}} CU<extra></extra>",
            )
        )
    layout(fig, "Loss cost per exposure-year: reported vs developed to ultimate", height=400)
    fig.update_xaxes(dtick=1, title="Policy year")
    fig.update_yaxes(title="CU per exposure-year")
    return fig


def fig_bi(eda: dict) -> go.Figure:
    fig = go.Figure()
    for i, (ay, r) in enumerate(sorted(eda["bi_sev"].items())):
        fig.add_trace(
            go.Scatter(
                x=list(r.keys()),
                y=list(r.values()),
                mode="lines+markers",
                name=f"Loss year {ay}",
                line={"color": S[i], "width": 2},
                marker={"size": 8},
                hovertemplate=f"{ay}, reported %{{x}} months after loss: mean %{{y:,.0f}} CU<extra></extra>",
            )
        )
    layout(fig, "Motor bodily injury: mean cost by reporting delay, by loss year", height=360)
    fig.update_xaxes(title="Months from loss to report")
    fig.update_yaxes(title="Mean net cost (capped at 250k), CU", tickformat=",.0f")
    return fig


def fig_contacts(eda: dict) -> go.Figure:
    c = eda["contacts"]
    fig = go.Figure()
    for i, (k, name) in enumerate(
        (
            ("dist_no_claim", "Train, no claim"),
            ("dist_with_claim", "Train, with a claim"),
            ("dist_test", "Test (2025)"),
        )
    ):
        d = c[k]
        fig.add_trace(
            go.Bar(
                x=[f"{int(float(x))}{'+' if float(x) >= 5 else ''}" for x in d],
                y=list(d.values()),
                name=name,
                marker={"color": S[i], "line": {"color": SURF, "width": 2}},
                hovertemplate=f"{name}<br>%{{x}} contacts: %{{y:.1%}} of rows<extra></extra>",
            )
        )
    layout(fig, "The leak: customer service contacts by claim status", height=340, barmode="group")
    fig.update_xaxes(title="Contacts in the period")
    fig.update_yaxes(title="Share of rows", tickformat=".0%")
    return fig


def fig_delay(eda: dict) -> go.Figure:
    fig = go.Figure()
    for i, (k, v) in enumerate(eda["delay_cdf"].items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(v))),
                y=v,
                mode="lines",
                name=k.replace("_", " "),
                line={"color": S[i % 8], "width": 2},
                hovertemplate=f"{k}<br>%{{x}} months: %{{y:.1%}} of ultimate reported<extra></extra>",
            )
        )
    layout(fig, "How fast claim cost gets reported, by line and peril group", height=380)
    fig.update_xaxes(title="Months after the loss month")
    fig.update_yaxes(title="Share of ultimate cost reported", tickformat=".0%")
    return fig


def fig_missing(eda: dict) -> go.Figure:
    m = pd.DataFrame(eda["missing"]).T
    m = m.drop(index=[i for i in m.index if "telematics_score" in i]).sort_values("train").tail(14)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=m.index,
            x=m["train"],
            orientation="h",
            name="Train",
            marker={"color": S[0]},
            hovertemplate="%{y}<br>Train missing %{x:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=m.index,
            x=m["test"],
            orientation="h",
            name="Test",
            marker={"color": S[1]},
            hovertemplate="%{y}<br>Test missing %{x:.1%}<extra></extra>",
        )
    )
    layout(
        fig,
        "Missing values: training vs test (telematics score excluded)",
        height=460,
        barmode="group",
    )
    fig.update_xaxes(title="Share of rows missing", tickformat=".0%")
    return fig


# --------------------------------------------------------------------------- page


CSS = """
:root{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:#0d366b;color:#fff;padding:28px 40px}header h1{margin:0 0 4px;font-size:24px}header p{margin:0;color:#cde2fb}
nav{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--grid);padding:8px 40px;display:flex;flex-wrap:wrap;gap:14px;font-size:13px}
nav a{color:var(--ink2);text-decoration:none}nav a:hover{color:var(--accent)}
main{max-width:1180px;margin:0 auto;padding:24px 40px 80px}
section{background:var(--surface);border:1px solid rgba(11,11,11,.08);border-radius:10px;padding:22px 26px;margin:22px 0}
h2{margin:0 0 10px;font-size:20px}h3{font-size:16px;margin:18px 0 6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:8px 0 4px}
.tile{background:#fff;border:1px solid var(--grid);border-radius:8px;padding:12px 14px}.tile .v{font-size:26px;font-weight:600}.tile .l{font-size:12px;color:var(--ink2)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--grid);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600;background:#f4f3ef}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff;white-space:nowrap}
.illus{display:block;max-width:760px;width:100%;margin-left:auto;margin-right:auto;border:1px solid var(--grid);border-radius:8px;margin:8px 0}
.note{color:var(--ink2);font-size:13px}.callout{border-left:4px solid var(--accent);background:#eef4fc;padding:10px 14px;border-radius:4px;margin:10px 0}
details summary{cursor:pointer;color:var(--accent);margin:6px 0}
"""


GROUP_LABEL = {
    "driver": "Driver",
    "vehicle": "Vehicle",
    "property": "Property",
    "location_hazard": "Location and hazard",
    "weather": "Weather",
    "policy_admin": "Policy and premium",
    "customer_marketing": "Customer and marketing",
    "claim_history": "Claim history",
}


def fig_perm(study: dict) -> go.Figure:
    lines = list(study["lines"])
    fig = make_subplots(
        rows=1,
        cols=len(lines),
        subplot_titles=[ln.title() for ln in lines],
        horizontal_spacing=0.16,
    )
    for i, ln in enumerate(lines, start=1):
        perm = study["lines"][ln]["permutation"]
        top = sorted(perm.items(), key=lambda kv: kv[1], reverse=True)[:10][::-1]
        fig.add_trace(
            go.Bar(
                x=[v for _, v in top],
                y=[k.replace("_", " ") for k, _ in top],
                orientation="h",
                marker={"color": LINE_COLOR[ln], "line": {"color": SURF, "width": 2}},
                showlegend=False,
                hovertemplate="%{y}: D² falls by %{x:.4f} when shuffled<extra></extra>",
            ),
            row=1,
            col=i,
        )
    layout(
        fig,
        "Which variables the model relies on (fold B, drop in D² when the variable is shuffled)",
        height=420,
    )
    fig.update_xaxes(tickformat=".3f")
    fig.update_annotations(font={"size": 13, "color": INK})
    return fig


def fig_groups(study: dict) -> go.Figure:
    fig = go.Figure()
    groups = list(GROUP_LABEL)
    for ln, rep in study["lines"].items():
        drops = rep["group_drop"]
        g = [k for k in groups if k in drops]
        fig.add_trace(
            go.Bar(
                x=[GROUP_LABEL[k] for k in g],
                y=[drops[k] for k in g],
                name=ln.title(),
                marker={"color": LINE_COLOR[ln], "line": {"color": SURF, "width": 2}},
                hovertemplate=f"{ln.title()}<br>%{{x}}: %{{y:+.4f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": AXIS, "width": 1})
    layout(
        fig,
        "What each group of variables adds (fold B; below zero, the line scores better without it)",
        height=380,
        barmode="group",
    )
    fig.update_yaxes(title="D² lost without the group", tickformat=".3f")
    return fig


LINE_NAMES = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents"}

REJECTED = """
| Engineered or tested predictor | Result (validation score) | Run |
|---|---|---|
| Peril-mechanics interactions (subsidence, escape of water, flood) | 0.1331, below the model then in use | ledger 2026-10-18 10:01 |
| Written premium ÷ line-year mean (drift-free premium) | 0.1336, inside noise | 10-18 11:09 |
| Lines a customer holds (cross-line) | 0.1330, no gain | 10-18 11:39 |
| Exposure as a predictor (log exposure) | 0.1309, worse | run 40 |
| Written premium ÷ sum insured | 0.1327, worse | run 42 |
| Property interaction bundle (sum insured bands, construction × age, flood × rainfall, soil × dry summer, excess ÷ sum insured) | worse than dropping premium | features review, 19 Oct |
| Weather split into address climate mean plus the year's anomaly | 0.1390, inside noise | 10-19 01:48 |
| Duplicate ages and build year removed | 0.1392, inside noise | 10-19 01:27 |
| Customer claim history split into same-line and other-line | crashed on 19 Oct; queued again | backlog 2.45 |
"""


def models_section(root: Path) -> str:
    path = root / "data" / "cache" / "model_card.json"
    if not path.exists():
        path = root / "reports" / "features" / "model_card.json"
    if not path.exists():
        return md("_The model card is being computed (scripts/model_card.py)._")
    card = json.loads(path.read_text())
    out = [
        md(f"""Each line has its own model. Predictors enter as recorded, except that categories with more than 15 levels (vehicle make and model, occupation, previous insurer) are replaced by a credibility-weighted average cost for the level, fitted out of fold. Numbers below come from the current model fitted on 2021-23 and scored on 2024 (fold B); pooled D² {card["pooled_d2"]:.4f}.

**How to read the lift columns.** Each predictor is shuffled across the 2024 policies and the line's predictions are rebuilt through the same models. *D² lost* is how much of the line's predictive power disappears without that predictor's information; *share* is that loss as a percentage of the line's D². The line's total level is fixed, so this measures how well the predictor separates cheap from expensive risks. A negative value means shuffling did not hurt: the predictor adds nothing measurable. Predictors that carry the same information (customer age and policyholder age, for example) share credit, so each can look small while the pair matters.""")
    ]
    for line, info in card["lines"].items():
        feats = info["features"]
        motor = line == "motor"
        head = "<tr><th>#</th><th>Predictor</th><th>D² lost</th><th>Share of line D²</th><th>Gini lost</th>"
        head += "<th>BI frequency part</th><th>Other perils part</th>" if motor else ""
        head += "</tr>"
        rows = []
        for i, f in enumerate(feats, 1):
            name = html.escape(f["feature"]) + (
                " <span class=note>(credibility-encoded)</span>" if f.get("encoded") else ""
            )
            r = (
                f"<tr><td class=num>{i}</td><td>{name}</td><td class=num>{f['d2_drop']:+.4f}</td>"
                f"<td class=num>{f['d2_drop'] / info['d2']:+.1%}</td><td class=num>{f['gini_drop']:+.4f}</td>"
            )
            if motor:
                r += f"<td class=num>{f.get('d2_drop_bi', float('nan')):+.4f}</td><td class=num>{f.get('d2_drop_other', float('nan')):+.4f}</td>"
            rows.append(r + "</tr>")
        useful = sum(f["d2_drop"] > 0.0005 for f in feats)
        out.append(
            f"<h3>{LINE_NAMES[line]}: {html.escape(info['components'])}</h3>"
            + md(
                f"Within-line D² {info['d2']:.4f}, Gini {info['gini']:.3f}. {info['n_features']} predictors; {useful} each lose more than 0.0005 of D² when shuffled."
            )
            + f"<table>{head}{''.join(rows)}</table>"
        )
        if info.get("ebm_terms"):
            terms = info["ebm_terms"][:20]
            trs = "".join(
                f"<tr><td>{html.escape(t['term'])}</td><td>{'interaction' if ' & ' in t['term'] else 'main effect'}</td><td class=num>{t['importance']:.3f}</td></tr>"
                for t in terms
            )
            out.append(
                md(
                    f"**{LINE_NAMES[line]} EBM, top 20 terms** (the EBM is additive, so each term is a curve that can be read directly; importance is the average absolute change it makes to the predicted rate; pairwise interactions are switched off for run time, so all terms are main effects):"
                )
                + f"<table><tr><th>Term</th><th>Type</th><th>Importance</th></tr>{trs}</table>"
            )
    out.append(
        md(
            """**Points for the actuary.** (1) Each line has a long tail of predictors that add nothing measurable on 2024 (negative or near-zero values above); they are candidates for pruning, which would make the tariff shorter without costing accuracy. (2) Several insurer-side fields still reach the models: campaign code, marketing segment, branch, sales channel and days from quote to inception. They describe how the policy was sold, not the risk. Their measured contribution is at most 0.001 on any line. Like written premium, they are an admissibility decision for the owner. (3) Credit band, occupation, marital status and customer age are in the models with small contributions (at most 0.001); they are flagged for the fairness review."""
        )
    )
    unused = card.get("unused", [])
    trs = "".join(
        f"<tr><td>{html.escape(u['column'])}</td><td>{html.escape(', '.join(u['lines']))}</td><td>{html.escape(u['reason'])}</td></tr>"
        for u in unused
        if not u["column"].startswith(("cost_", "n_", "ult_", "comp_"))
    )
    out.append(
        "<h3>Available but not used</h3>"
        + md(
            "Every column in the data that does not reach a line's model, and why. Claim outcome columns for the row (counts, costs and developed costs by peril) are the target side and are always hidden; they are not listed."
        )
        + f"<table><tr><th>Column</th><th>Lines</th><th>Reason</th></tr>{trs}</table>"
        + "<h3>Engineered predictors tested and not adopted</h3>"
        + md(REJECTED)
    )
    return "".join(out)


SUBMODEL_NAMES = {
    "motor_bi": "Motor: bodily injury claim frequency",
    "motor_other": "Motor: other perils",
    "buildings": "Buildings",
    "contents": "Contents",
}


def _lst(v):
    return json.loads(v.replace("'", '"')) if isinstance(v, str) else v


def interactions_section(root: Path) -> str:
    tw_path = root / "reports" / "features" / "twoway.json"
    it_path = root / "reports" / "features" / "interactions.json"
    if not tw_path.exists():
        return md("_Two-way tables not yet computed._")
    tw = json.loads(tw_path.read_text())
    it = json.loads(it_path.read_text()) if it_path.exists() else {}
    out = [
        md("""**How the modelling has handled interactions so far.** Gradient-boosted trees capture interactions implicitly: a split on one predictor inside a split on another. The EBM member on property runs main effects only. Hand-built interaction features were tried twice (peril-mechanics terms on 18 October, a property bundle on 19 October) and again on the detected pairs below; each time they scored worse. Square-root, log or square transforms of a single predictor make no difference to a tree model, which uses only the ordering of values; they matter for GLMs and smooth curves.

**How interactions were looked for.** Three methods: the EBM's own interaction detector, Friedman's H-statistic from a boosted model, and residual actual-to-expected tables against the current model's 2024 predictions. All three put flood zone × elevation first on property; motor's interactions are about a hundredth as strong. The heatmaps show observed loss cost (bodily injury: claim frequency) on 2021-23 for the strongest pairs, as a relativity to the line average; where available, the label adds the current model's actual-to-expected on 2024.""")
    ]
    for sub, pairs in tw.items():
        ranked = [tuple(r["pair"]) for r in it.get(f"{sub}_hstat", [])]
        order = {frozenset(pr): i for i, pr in enumerate(ranked)}
        pairs = sorted(pairs, key=lambda q: order.get(frozenset((q["x"], q["y"])), 999))[:4]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"{q['x']} × {q['y']}" for q in pairs],
            horizontal_spacing=0.18,
            vertical_spacing=0.22,
        )
        for k, q in enumerate(pairs):
            act = np.array(_lst(q["actual"]), dtype=float)
            ex = np.array(_lst(q["exposure"]), dtype=float)
            ae = np.array(_lst(q["ae"]), dtype=float)
            mean = np.nansum(act * ex) / np.nansum(ex)
            rel = act / mean

            def cell(i, j):
                if not np.isfinite(rel[i, j]) or ex[i, j] <= 0:
                    return ""
                lab = f"{rel[i, j]:.2f}"
                if sub != "motor_bi" and np.isfinite(ae[i, j]) and ae[i, j] > 0:
                    lab += f"<br>A/E {ae[i, j]:.2f}"
                return lab

            text = [[cell(i, j) for j in range(rel.shape[1])] for i in range(rel.shape[0])]
            fig.add_trace(
                go.Heatmap(
                    z=rel,
                    x=_lst(q["y_levels"]),
                    y=_lst(q["x_levels"]),
                    text=text,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                    hovertemplate=f"{q['x']} %{{y}}<br>{q['y']} %{{x}}<br>relativity %{{z:.2f}}<extra></extra>",
                ),
                row=k // 2 + 1,
                col=k % 2 + 1,
            )
        layout(fig, SUBMODEL_NAMES.get(sub, sub), height=640, showlegend=False)
        fig.update_xaxes(tickfont={"size": 9})
        fig.update_yaxes(tickfont={"size": 9})
        out.append(chart(fig))
    rows = []
    for sub in ("buildings", "contents", "motor_other", "motor_bi"):
        h = it.get(f"{sub}_hstat", [])[:3]
        rows.append(
            f"| {SUBMODEL_NAMES[sub]} | "
            + "; ".join(f"{a} × {b} ({r['h_stat']:.2f})" for r in h for a, b in [r["pair"]])
            + " |"
        )
    out.append(
        md(
            "**Strongest pairs by H-statistic** (share of the pair's joint effect that is not additive; 0 = purely additive):\n\n| Sub-model | Top pairs |\n|---|---|\n"
            + "\n".join(rows)
        )
    )
    out.append(
        md("""**Techniques compared on the full admissible predictor set** (light single-seed fits on 2024, change in within-line D² against the current LightGBM approach; measured noise 0.0012 buildings, 0.0006 contents):

| Technique | Motor | Buildings | Contents |
|---|---|---|---|
| CatBoost | n/a | +0.0077 | +0.0037 |
| EBM with 15 interactions | −0.032 | +0.0048 | +0.0044 |
| Spline GLM (transparent benchmark) | +0.0075 | −0.013 | −0.017 |
| Explicit interaction features | worse | worse | worse |
| All valid predictors instead of the pruned set | worse | worse | worse |

The three gains are queued for the loop as full tests (backlog 2.70, 2.71, 2.73). Full review: reports/reviews/2026-10-19_interactions.md.""")
    )
    return "".join(out)


def fairness_section(root: Path) -> str:
    path = root / "reports" / "reviews" / "2026-10-20_fairness.md"
    if not path.exists():
        return md("The fairness review has not run yet.")
    text = path.read_text()
    rec = text.split("## 5. Recommendation", 1)[1].split("## Scripts", 1)[0]
    dfp = text.split("## 4. Discrimination-free pricing check", 1)[1].split("## 5.", 1)[0]
    return md(
        """For each sensitive field the review measured three things on fold B: how much accuracy the field adds, how well the model's other inputs rebuild it (proxy strength), and how prices move under three versions of the model: with the field, refitted without it, and discrimination-free (the field averaged out over the portfolio, after Lindholm, Richman, Tsanakas and Wüthrich, 2022). Sex, race, ethnicity and religion are not in the data. The review used a light single-seed model, so differences under about 0.002 D² are noise.

The one change it proposed, dropping the duplicate age column on contents, was tested by the loop at 09:31 on 20 October: 0.14263 against 0.14260, neutral, so the model is unchanged.

### Recommendation by field"""
        + rec
        + "\n### Discrimination-free check"
        + dfp
        + "\nFull review: reports/reviews/2026-10-20_fairness.md."
    )


def explain_section(root: Path, assets: Path) -> str:
    path = root / "reports" / "explain" / "explain.json"
    if not path.exists():
        return md("_Explainability pack not yet run._")
    ex = json.loads(path.read_text())
    out = [
        md("""**Why this section.** A regulator, a pricing committee or a judge needs to see that each rating factor moves the price in a direction that makes sense, by a believable amount, and consistently. Three checks go beyond actual-to-expected:

1. **Relativity curves** for every predictor in every sub-model: the predicted rate across the predictor's levels with everything else held at its observed values, set to 1.00 at a base level. Two versions: partial dependence (the familiar one) and accumulated local effects (ALE), which stays honest when predictors are correlated (driver age and years licensed, flood zone and elevation). The observed one-way experience on 2021-23 is shown alongside.
2. **Direction checks.** An expected direction for each predictor from actuarial reasoning (no-claims years and excess down; mileage, sum insured, flood zone and prior claims up; driver age U-shaped), then for each: the model's direction, the share of individual policies whose own curve goes the other way (ICE), whether the 2021-22 and 2021-23 fits agree, and flags for contradictions or wiggles a regulator would query.
3. **A surrogate GLM** fitted to the model's own predictions per line: a conventional relativity table that shows how much of the model a GLM can reproduce.""")
    ]
    sur = ex.get("surrogate", {})
    if sur:
        rows = "\n".join(
            f"| {ln.title()} | {s['d2_tweedie']:.2f} | {s['r2_weighted']:.2f} | {len(s['features'])} |"
            for ln, s in sur.items()
        )
        out.append(
            md(f"""**Surrogate GLM.** A banded Tweedie GLM on 12 predictors per line reproduces most of what the model does:

| Line | Share of model's deviance explained (D²) | R² on the rate | Predictors |
|---|---|---|---|
{rows}

Its relativities line up with the curves below. It is the bridge for a filing: the GLM table explains the bulk; the boosted model adds the rest, chiefly through interactions such as flood zone × elevation.""")
        )
    flags = []
    for ln, L in ex["fold_B"]["lines"].items():
        for sm, S in L["sub_models"].items():
            for col, P in S["predictors"].items():
                real = [f for f in P.get("flags", []) if "wiggle" not in f]
                if real:
                    flags.append((ln, sm, col, "; ".join(real)))
    trs = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td><td>{html.escape(d)}</td></tr>"
        for a, b, c, d in flags
    )
    out.append(
        md(f"""**Direction check results.** {sum(len(S["predictors"]) for L in ex["fold_B"]["lines"].values() for S in L["sub_models"].values())} predictor × sub-model combinations were tested. Flags that are only "wiggles" on an 8-point grid are omitted here (they mean "look at the chart", not "wrong"). The substantive flags:

**Points for the actuary.** (1) Wildfire index runs against its expected upward direction on motor other perils and contents, and rainfall against it on buildings: candidates to constrain monotone or remove. (2) Distance to river on motor and buildings, and customer claim history on buildings, also cut against the prior. (3) Not testable by a sign check: monotone-constrained fields (their direction is enforced), ordinal fields stored as categories (credit band, security level, urban density) and U-shaped priors (driver age, policyholder age); read their charts.""")
        + f"<table><tr><th>Line</th><th>Sub-model</th><th>Predictor</th><th>Flag</th></tr>{trs}</table>"
    )
    pngs = sorted(assets.glob("explain_*.png"))
    if pngs:
        out.append(
            md(
                "**Relativity charts for the main predictors** (partial dependence and ALE against the observed one-way, base level = 1.00):"
            )
        )
        out.append(
            '<div class="grid2">'
            + "".join(img(q, q.stem.replace("explain_", "").replace("_", " ")) for q in pngs)
            + "</div>"
        )
    lit = root / "reports" / "explain" / "literature.md"
    if lit.exists():
        out.append(
            "<details><summary>Literature behind these methods (Richman and Wüthrich; Laval; Utrecht; ALE and SHAP)</summary>"
            + md(lit.read_text())
            + "</details>"
        )
    return "".join(out)


def section(sid: str, title: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{html.escape(title)}</h2>{body}</section>'


def build(root: Path, out: Path, batch: str, assets: Path) -> None:
    df = load_ledger(root)
    eda = json.loads((assets / "eda.json").read_text())
    preds = root / "data" / "cache" / "preds"
    kept = df[df["verdict"].isin(["keep", "keep_owner"])]
    best = kept.loc[kept["score"].idxmax()] if not kept.empty else None
    base = df.iloc[0]
    subs = [
        ln
        for ln in (root / "docs" / "SIGNOFF.md").read_text().splitlines()
        if re.match(r"\| 20\d\d-", ln)
    ]
    lb = [float(m.group(1)) for ln in subs if (m := re.search(r"\| (0\.\d{4,5}) \|", ln))]
    n_runs = len(df) - 1

    tiles = f"""<div class="tiles">
<div class="tile"><div class="v">{best["score"]:.4f}</div><div class="l">Best validation score (run {best["n"]})</div></div>
<div class="tile"><div class="v">{max(lb):.4f}</div><div class="l">Best public leaderboard score</div></div>
<div class="tile"><div class="v">{n_runs}</div><div class="l">Experiments after the baseline</div></div>
<div class="tile"><div class="v">{(df["verdict"].isin(["keep", "keep_owner"])).sum() - 1}</div><div class="l">Kept improvements</div></div>
<div class="tile"><div class="v">{len(subs)}</div><div class="l">Leaderboard submissions</div></div>
</div>"""

    summary = tiles + md(f"""
- The model has improved from a validation score of **{base["score"]:.4f}** to **{best["score"]:.4f}** over {n_runs} supervised experiments. On the public leaderboard it moved from 0.1060 to {max(lb):.4f} (13th on 20 October; the leader is at about 0.130).
- **Current model:** motor bodily injury is priced on claim counts (LightGBM Poisson) with a separate severity level; other motor perils blend LightGBM with an Explainable Boosting Machine (EBM, a glass-box additive model). Buildings and contents average three learners: LightGBM, an EBM and CatBoost, with the number of boosting rounds set by early stopping on an inner time split. Each line's level is anchored to its trended history.
- **Owner decisions** (each scored and logged in the assumptions log): email domain, written premium on every line, licence points on property and five sales fields removed (channel kept); the 11.9m motor bodily injury claim capped at 1m with the excess loaded from a fitted Pareto tail; bodily injury developed by frequency x severity without the "larger of two estimates" floor (20 October, +0.0002); buildings trended by peril (weather and escape of water 9%, fire and accidental damage 4%; 20 October, +0.0002).
- **Fairness (20 October):** age, credit band and occupation (buildings) stay in; marital status and household size stay out. Dropping age alone would change nothing, because years licensed rebuilds it almost exactly; a discrimination-free price would cost about 10% of motor accuracy. See the Fairness section.
- **The ideas list is exhausted on the current target.** Every motor item, including nine from an independent second-opinion review, was tried on 20 October and none cleared the bar. Remaining gains are expected from the target (development and trend choices), not the model; see What comes next.
- Local validation predicts the leaderboard well on the later validation year (fold B) and overstates gains on the earlier one (fold A). The score weights fold B twice for that reason, and the final submission is the best fold B candidate (rule pre-declared in docs/SIGNOFF.md). Why two out-of-time folds, and what they cannot test, is question Q19.
- Data problems found and treated: exposure recorded in days on about 0.4% of rows, a field that leaks claim information in the training data only, light motor bodily injury case reserves, and the 11.9m claim. Orphan claims (loss before the policy's first row) are 0.23% of cost and immaterial.
- An AI agent proposes one change at a time; a frozen harness scores it; a fixed rule keeps or reverts it, and a candidate keep must pass again on fresh seeds; nothing reaches the leaderboard without the actuary's sign-off.
""")

    evals = md(f"""
| Measure | What it tells you | How it is used |
|---|---|---|
| **Tweedie D² (power 1.5)** | Share of the deviance of a flat "everyone pays the average" model that our model removes. 0 means no better than the average; higher is better. Tweedie deviance suits insurance costs: many zeros and a long right tail. | The leaderboard metric. Computed locally exactly as Kaggle does (confirmed by a calibration submission). |
| **Score (weighted)** | Fold A and fold B D², with fold B counted twice. | The number the loop optimises. Fold B is weighted up because it tracks the leaderboard. |
| **Worse fold** | The lower of the two fold scores. | A gate: a change that improves the average but hurts either year by more than seed noise is rejected. |
| **Noise floor ({NOISE_FLOOR})** | How much the score moves from random seeds alone: standard deviation 0.0002 for the seed-bagged model (measured 19 October over five full seed shifts). The floor was 0.002 until 19 October, 0.001 on 19 October and 0.0005 from 20 October, each lowered by the owner after the seed-noise measurements. | A change must beat the best model by more than this to be kept. |
| **Within-line D²** | Accuracy of the ranking inside motor, buildings and contents separately. | Shows where improvement is coming from; pooled D² also rewards getting line levels right. |
| **Actual ÷ expected by line** | Whether each line's total is priced at the right level. | Should sit between 0.97 and 1.03. |
| **Lift and Gini** | How well the model separates cheap risks from expensive ones; the lift chart shows actual cost by predicted-risk decile. | A steeper lift curve means better risk selection. |
""")

    folds_txt = md(f"""
The training data covers policy years 2021 to 2024 ({eda["exposure_fix"]["train_rows"]:,} policy-period rows); the test set is 2025 ({eda["exposure_fix"]["test_rows"]:,} rows). Because the task is to forecast a future year, validation is **out of time**: the model is trained on earlier years and scored on the next one, twice. Random cross-validation would mix years and overstate accuracy.

Every policy-period row is used; nothing is sampled away. A policy can appear in several years, which mirrors 2025 (85% of 2025 policies were insured before), so rows are not grouped by policy. Training targets are developed to ultimate and trended to the year being predicted; each line's overall level is anchored to the trended history, and the model supplies the relativities.
""")

    ly = pd.DataFrame(eda["line_year"])
    t = ly.assign(freq=lambda d: d.claims / d.exposure, lc=lambda d: d.ultimate / d.exposure)
    data_tbl = (
        "<table><tr><th>Line</th><th>Year</th><th class=num>Rows</th><th class=num>Exposure-years</th><th class=num>Claims</th><th class=num>Frequency</th><th class=num>Developed loss cost (CU)</th></tr>"
        + "".join(
            f"<tr><td>{r.line.title()}</td><td>{r.year}</td><td class=num>{r.rows:,.0f}</td><td class=num>{r.exposure:,.0f}</td><td class=num>{r.claims:,.0f}</td><td class=num>{r.freq:.1%}</td><td class=num>{r.lc:,.0f}</td></tr>"
            for r in t.itertuples()
        )
        + "</table>"
    )
    dq = (root / "reports" / "data_quality.md").read_text().split("\n", 2)[2]

    data_body = (
        md("### What the data is")
        + data_tbl
        + md("### Sample and validation approach")
        + folds_txt
        + img(assets / "folds.png", "Out-of-time validation folds")
        + md("### Data cleaning: what was checked and what was done")
        + md(dq)
        + md("### Exploratory analysis")
        + '<div class="grid2">'
        + chart(fig_contacts(eda))
        + chart(fig_missing(eda))
        + "</div>"
        + md(
            f"""<p class="note">The contacts chart is the clearest finding: training rows with a claim show 3 to 5 contacts, rows without a claim mostly 0 or 1, and the 2025 test rows look exactly like the no-claim rows ({eda["contacts"]["test"]:.2f} vs {eda["contacts"]["no_claim"]:.2f} per exposure-year). The field records claim calls in training only, so it would look like the best predictor and then fail on 2025. It is excluded. The missing-values chart leaves out telematics score, which is blank for the 88% of motor policies that did not opt in to telematics; that is by design, not a defect. The other gaps are about 2.5% everywhere, in training and test alike, which points to deliberately injected noise rather than a pattern.</p>"""
        )
    )

    ult_body = (
        img(assets / "ultimate.png", "From reported claims to ultimate cost")
        + md(
            """The leaderboard scores predictions against **ultimate** cost, so training targets must be ultimate too. Claims are valued once, at 31 December 2024, and 2024 losses are a third unpaid. Development uses a chain ladder on reported cost by loss month and reporting delay, for each line and peril group. Motor bodily injury reports slowly (7% of cost in the loss month, 85% within a year) and its open case reserves are too light: for the same reporting delay, younger loss years show lower average cost while every other peril inflates. It is therefore projected as ultimate claim count times the 2021 average cost, trended at 3.5% a year."""
        )
        + '<div class="grid2">'
        + chart(fig_delay(eda))
        + chart(fig_bi(eda))
        + "</div>"
        + chart(fig_losscost(eda))
        + md(
            '<p class="note">The test of the method: developed 2024 should sit on the 2021-2023 trend. Raw motor 2024 was 12% short; developed, it is within 2%. Buildings 2024 sits above trend, consistent with a weather-heavy year. This test runs automatically before every experiment.</p>'
        )
    )

    lift, spread = fig_lift(df, preds)
    progress_body = (
        img(assets / "loop.png", "Supervised experiment loop")
        + chart(fig_progress(df))
        + md(
            '<p class="note">The line is the best kept model. A discarded run can sit slightly above it: its gain was inside the noise floor (0.002 until 19 October, 0.001 on 19 October, 0.0005 since) or it hurt one validation year, so the rule did not count it. Purple marks owner decisions.</p>'
        )
        + '<div class="grid2">'
        + chart(fig_folds(df))
        + (chart(fig_cv_lb(root)) if fig_cv_lb(root) else "")
        + "</div>"
        + '<div class="grid2">'
        + chart(fig_lines(df))
        + chart(fig_ae(df))
        + "</div>"
        + (
            chart(lift) + chart(spread)
            if lift
            else md("_Lift charts appear as runs save their predictions._")
        )
        + md(
            '<p class="note">Earlier kept runs were re-scored from their exact code in git to fill in their lift curves. The curves overlap closely because each change is small; the separation bars show the steady gain.</p>'
        )
        + md("### Approaches compared")
        + chart(fig_family(df))
    )

    rows_html = "".join(
        f"<tr><td class=num>{r.n}</td><td><span class=badge style='background:{VERDICT.get(r.verdict, (MUTED, r.verdict))[0]}'>{VERDICT.get(r.verdict, (MUTED, r.verdict))[1]}</span></td>"
        f"<td class=num>{'' if pd.isna(r.score) else f'{r.score:.4f}'}</td><td class=num>{'' if pd.isna(r.delta) else f'{r.delta:+.4f}'}</td>"
        f"<td>{html.escape(r.family)}</td><td>{html.escape(r.why)}</td><td>{html.escape(str(r.description))}</td></tr>"
        for r in df.itertuples()
    )
    tested = (
        "<table><tr><th>Run</th><th>Outcome</th><th>Score</th><th>Change vs best</th><th>Approach</th><th>Why</th><th>Hypothesis (as written by the agent or owner)</th></tr>"
        + rows_html
        + "</table>"
    )

    backlog = (root / "research" / "experiment_backlog.md").read_text().split("\n", 2)[2]
    alternatives = md("""
Beyond the queue, four broad directions frame the comparisons above:

| Direction | Options | Current position |
|---|---|---|
| Target structure | One direct cost model per line, or frequency × severity by peril | Direct model; peril split is next (1.1, 1.2) |
| Level | Anchored to trended history, floating with the model, or weather perils floating | Anchored; weather float is next |
| Learner | GLM, gradient boosting, GLM with a boosted correction (CANN), neural networks | Gradient boosting; GLM reference and CANN queued; neural nets on the GPU track |
| Development method | Chain ladder on reporting delay, frequency × severity for bodily injury, reporting-delay model, Cape Cod / Bornhuetter-Ferguson | First two in use; others on the owner track |

The GPU track (TabM, a neural network for tables that Richman's 2026 paper found scales best on motor pricing data) runs only with explicit approval and a run-time and quota estimate.
""")

    research = md("""
| Stream | Main findings | What it changed |
|---|---|---|
| Kaggle winners | Train on the loss the metric uses; seed-average; encode categories within folds; blend a few diverse models; trust local validation over the public board | Training power, seed bagging, submission discipline |
| Jurisdiction | Reads as England and Wales (inference): unlimited injury cover, Ogden rate, periodical payment orders, 2021 whiplash tariff | Bodily injury treated as its own problem; capping from own data |
| Richman and co-authors | LocalGLMnet, ICEnet, Credibility Transformer, nagging predictors, discrimination-free pricing; CANN is Wüthrich & Merz and isotonic recalibration is Wüthrich & Ziegel | Credibility encodings, CANN, calibration and wider bagging queued |
| Public code | Usable: RonRichman/frmtpl-scaling-laws (GPL-2.0), lorentzenchr/credibility (Apache-2.0), model-diagnostics (MIT). Not usable: Credibility Transformer code (non-commercial licence), course notebooks without a licence | GPU track built on the scaling-laws code |
| Models on Hugging Face | No insurance-specific models. TabPFN fails the licence rule and is built for 50k rows; TabICL, Mitra, CARTE are licence-clean but not built for 2.4M rows | Only TabM and RealMLP are worth a slot |
| Reserving (CAS, IFoA) | With one valuation date: reporting-delay models with right truncation, Cape Cod / BF with a pricing prior, a cross-sectional case-adequacy check; backtest reporting patterns as if valued at end-2023 | Owner track 3.2 to 3.5 |
| Feature engineering | Out-of-fold credibility encoding for high-cardinality fields; hand-built peril interactions; null-importance selection; no binning needed for trees; predictor outliers matter little to trees | Tier 2 items |
| Standards | CIA Standards of Practice 1440, 1450, 1490, 1700; IAA AI Task Force papers; CAS AI Primer (agentic AI) | docs/governance.md |

**Feature treatment so far, plainly:** raw rating factors go into the boosted trees; missing values are left for the trees to handle; nothing is binned (trees split continuous values themselves); predictor outliers are not trimmed; no interactions are engineered yet; one leaking field and two drifting fields are excluded. Tier 2 of the queue addresses the gaps.
""")

    compute = md("""
**Time budget per experiment:** 30 minutes. It keeps the loop at two to five experiments an hour and stops a runaway run using up the night. With the EBM in the model a run takes 13 to 18 minutes, plus a 2-minute crash check, plus a fresh-seed rerun for any candidate keep. Slow models that add variety (TabM, possibly others) are fitted once outside the loop and blended afterwards instead of raising the limit.

**CPU:** overnight the machine was 40 to 50% idle because the EBM uses one core per internal bag and was set to two bags. Raising it to eight is first in the queue.

**GPU:** TabM ran on the Mac's own GPU (Apple Metal). A like-for-like benchmark on a Kaggle Tesla T4 (synthetic data shaped like the motor features, about 2 GPU minutes) ran 5.9 seconds per epoch per 100,000 rows against 14.5 on the Mac when idle: about 2.5 times faster. A full TabM refit would take about 30 minutes on Kaggle against about 80 locally, but the dataset would first have to be rebuilt there. No other Kaggle GPU time has been used.
""")

    questions = (root / "docs" / "questions.md").read_text().split("\n", 2)[2]
    gov = img(assets / "governance.png", "Governance layers") + md(
        """Every artefact maps to a CIA standard: data checks (1440), the frozen harness and validation (1450), the ledger, logs and assumptions record (1490), and the report structure (1700). The IAA AI Task Force papers on governance, testing and documentation set the structure; the CAS AI Primer covers agent risks. Full map: `docs/governance.md`."""
    )

    study_path = root / "reports" / "features" / "feature_study.json"
    variables = md("The variable study has not run yet.")
    if study_path.exists():
        study = json.loads(study_path.read_text())
        rows = []
        for ln, rep in study["lines"].items():
            ow = rep["one_way"]
            drift = sorted(
                ((k, v.get("drift_test_sd", 0.0)) for k, v in ow.items()),
                key=lambda kv: -abs(kv[1]),
            )[:3]
            inter = list(rep["interactions"])[:3]
            rows.append(
                f"| {ln.title()} | {rep['d2_full']:.3f} | {len(ow)} | "
                + ", ".join(f"{k.replace('_', ' ')} ({v:+.2f})" for k, v in drift)
                + " | "
                + "; ".join(i.replace("_", " ") for i in inter)
                + " |"
            )
        variables = (
            md(f"""The study asks three questions of every candidate rating variable, line by line, using the later validation year (train 2021-2023, score 2024). It ran on {study["generated"]}.

1. **On its own:** how cost varies across the variable's levels, whether that pattern held in 2024, and how far the 2025 test data has drifted from training.
2. **Inside the model:** how much accuracy is lost if the variable's values are shuffled (permutation importance), and how much each variable moves individual predictions (SHAP).
3. **In combination:** how much accuracy is lost when a whole group of related variables is removed and the model refitted, and which pairs of variables interact most.""")
            + chart(fig_perm(study))
            + chart(fig_groups(study))
            + md(
                "| Line | Study model D² | Variables | Largest train-to-test drift (standard deviations) | Strongest interactions |\n|---|---|---|---|---|\n"
                + "\n".join(rows)
            )
            + md("""**What it shows, plainly**

- **Written premium is the strongest single variable on every line.** It carries the previous insurer's view of the risk. It also drifts: contents premiums in 2025 sit 0.4 standard deviations above training. A tree that splits on absolute premium amounts will misplace 2025 policies, so the queue tests premium as a relativity within each year (item 2.19).
- **Motor** is driven by the driver: mileage, no-claims years, years licensed and age. Removing the driver group costs 0.011 of D², more than any other group.
- **Buildings and contents** are driven by location hazard (elevation, flood zone, distance to river) and size (sum insured). Flood zone with elevation is the strongest interaction in contents.
- **Some groups cost accuracy.** Without the vehicle group, motor scores 0.006 better. Without the customer and marketing fields, buildings and contents each score 0.0035 better. The study model is simpler than the production model (it uses LightGBM's own category handling, not credibility encoding), so these are leads, not conclusions. The loop now tests them under the full harness (items 2.17 and 2.18).

<p class="note">The study model is one fit per line without seed averaging, so differences under about 0.002 are noise. Full output: reports/features/feature_study.json.</p>

**Variables that need an actuarial decision, not just a score.** Customer age, marital status, occupation, email domain and credit band may predict cost but raise fairness, regulatory or public-acceptability questions. Age and marital status are protected or restricted rating factors in some jurisdictions. These are flagged for the owner's review (stage 5 of the study), with a discrimination-free pricing check (Lindholm, Richman, Tsanakas and Wüthrich) as the method if any are kept.""")
        )

    nav = "".join(
        f'<a href="#{i}">{t}</a>'
        for i, t in [
            ("summary", "Summary"),
            ("evals", "How we measure"),
            ("data", "Data"),
            ("ultimate", "Ultimate cost"),
            ("progress", "Progress"),
            ("models", "Models"),
            ("variables", "Variables"),
            ("interactions", "Interactions"),
            ("explain", "Explainability"),
            ("fairness", "Fairness"),
            ("tested", "Ideas tested"),
            ("next", "Next"),
            ("alt", "Directions"),
            ("research", "Research"),
            ("questions", "Questions"),
            ("compute", "Compute"),
            ("governance", "Governance"),
        ]
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IAA hackathon: experiment dashboard</title><style>{CSS}</style><script src="assets/plotly.min.js"></script></head><body>
<header><h1>IAA AI-Assisted Actuarial Hackathon 2026: experiment dashboard</h1>
<p>Part 1 · risk premium for motor, buildings and contents · {batch} · generated {datetime.now():%Y-%m-%d %H:%M}</p></header>
<nav>{nav}</nav><main>
{section("summary", "Summary", summary)}
{section("evals", "How performance is measured", evals)}
{section("data", "The data, sample and validation approach", data_body)}
{section("ultimate", "Getting the target right: developing claims to ultimate", ult_body)}
{section("progress", "Experiment progress", progress_body)}
{section("models", "The models: predictors and the lift each adds", models_section(root))}
{section("variables", "Rating variables: individual and combined predictive power", variables)}
{section("interactions", "Interactions: two-way views and what the model captures", interactions_section(root))}
{section("explain", "Explainability: do the relativities make sense?", explain_section(root, assets))}
{section("fairness", "Fairness and proxy review", fairness_section(root))}
{section("tested", "Ideas tested so far", tested)}
{section("next", "What comes next, and why", md(backlog))}
{section("alt", "Alternative directions", alternatives)}
{section("research", "Research findings", research)}
{section("questions", "Questions and hypotheses register", md(questions))}
{section("compute", "Compute budget and GPU", compute)}
{section("governance", "Governance and professional standards", gov)}
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    js = out.parent / "assets" / "plotly.min.js"
    if not js.exists():
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(get_plotlyjs())
    out.write_text(page)
    if batch.startswith("after batch"):
        (out.parent / f"{batch.replace(' ', '_')}.html").write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parents[1]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--batch", default="")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "reports" / "dashboard" / "index.html"
    assets = Path(a.assets) if a.assets else root / "reports" / "dashboard" / "assets"
    batch = f"after batch {a.batch}" if a.batch else "snapshot"
    build(root, out, batch, assets)


if __name__ == "__main__":
    main()

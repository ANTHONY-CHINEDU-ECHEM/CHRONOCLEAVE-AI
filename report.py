"""Figures for the README, rebuilt from saved predictions and metrics."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from chronocleave.config import Config, resolve
from chronocleave.data.schema import EVENT_DESCRIPTIONS, EVENTS, T5_OPTIMAL, TARGET
from chronocleave.features.intervals import add_intervals

INK, INDIGO, ROSE, GOLD, GREY = "#14213D", "#3D348B", "#E63946", "#F7B801", "#8D99AE"
LABELS = {"rule_based": "Rule based windows", "logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}


def _style() -> None:
    plt.rcParams.update({"savefig.dpi": 160, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.grid": True, "grid.alpha": 0.25})


def _label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    kind, size = name.split("_")
    return f"{kind.upper()} ({size} units)"


def generate_figures(cfg: Config) -> list[str]:
    _style()
    report_dir, figure_dir = resolve(cfg.artifacts.report_dir), resolve(cfg.artifacts.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((report_dir / "metrics.json").read_text())
    preds = pd.read_csv(report_dir / "test_predictions.csv")
    raw = add_intervals(pd.read_csv(resolve(cfg.data.raw_path)))
    y, selected, results = preds[TARGET].to_numpy(), report["selected_model"], report["results"]
    best_lstm = max((k for k in results if k.startswith("lstm")), key=lambda k: results[k]["val"]["roc_auc"])
    best_tcn = max((k for k in results if k.startswith("tcn")), key=lambda k: results[k]["val"]["roc_auc"])
    written = []

    # 1. ROC curves.
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    for name, colour, style in [("rule_based", GREY, "--"), ("logistic_regression", GOLD, "-"), ("gradient_boosting", ROSE, "-"),
                                (best_lstm, INK, "-"), (best_tcn, INDIGO, "-")]:
        fpr, tpr, _ = roc_curve(y, preds[name])
        ax.plot(fpr, tpr, color=colour, ls=style, lw=2.3 if name == selected else 1.5,
                label=f"{_label(name)}  {results[name]['test']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=INK, lw=0.7, alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Implantation prediction on unseen patients")
    ax.legend(title="Test ROC AUC", frameon=False, loc="lower right", fontsize=9.5)
    fig.tight_layout(); fig.savefig(figure_dir / "roc_curves.png"); plt.close(fig); written.append("roc_curves.png")

    # 2. Model comparison across three views of performance.
    names = ["rule_based", "logistic_regression", "gradient_boosting", best_lstm, best_tcn]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("roc_auc", "ROC AUC, all transfers"), ("sibling_pairwise_accuracy", "Sibling ranking accuracy"), ("roc_auc_day3", "ROC AUC, day 3 transfers only")]
    for ax, (metric, title) in zip(axes, panels):
        values = [results[n]["test"][metric] for n in names]
        bars = ax.barh([_label(n) for n in names], values, color=[INDIGO if n == selected else GREY for n in names])
        for bar, value in zip(bars, values):
            ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9.5)
        ax.set(xlim=(0.5, max(values) + 0.04), title=title)
        ax.invert_yaxis()
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(figure_dir / "model_comparison.png"); plt.close(fig); written.append("model_comparison.png")

    # 3. What separates implanting embryos: interval distributions.
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    spec = [("t5", "t5 (hours post insemination)", (38, 70)), ("s2", "s2: t4 minus t3 (hours)", (0, 6)),
            ("cc2", "cc2: t3 minus t2 (hours)", (0, 18)), ("blastulation_duration", "tSB to tB (hours)", (2, 12))]
    for ax, (column, label, limits) in zip(axes, spec):
        bins = np.linspace(*limits, 40)
        for outcome, colour, name in [(0, GREY, "Did not implant"), (1, INDIGO, "Implanted")]:
            ax.hist(raw.loc[raw[TARGET] == outcome, column].dropna(), bins=bins, density=True, alpha=0.6, color=colour, label=name)
        if column == "t5":
            ax.axvspan(*T5_OPTIMAL, color=GOLD, alpha=0.18, lw=0)
        ax.set(xlabel=label, yticks=[])
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Timing profiles by outcome", loc="left")
    fig.tight_layout(); fig.savefig(figure_dir / "timing_distributions.png"); plt.close(fig); written.append("timing_distributions.png")

    # 4. Event occlusion importance.
    drops = report["selected_extras"]["event_occlusion_auc_drop"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    values = [drops[e] for e in EVENTS]
    ax.bar(EVENTS, values, color=[INDIGO if v > 0 else GREY for v in values])
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(7.5, color=INK, lw=0.8, ls=":")
    ax.text(7.6, max(values) * 0.95, "extended culture only", fontsize=9, va="top")
    ax.set(ylabel="Loss of test ROC AUC when hidden", title=f"Which annotations the {_label(selected)} model depends on")
    fig.tight_layout(); fig.savefig(figure_dir / "event_importance.png"); plt.close(fig); written.append("event_importance.png")

    # 5. Observed implantation by score decile.
    deciles = pd.qcut(preds[selected], 10, labels=False, duplicates="drop")
    table = preds.groupby(deciles).agg(observed=(TARGET, "mean"), predicted=(selected, "mean"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.index + 1, 100 * table["observed"], color=INDIGO, label="Observed implantation rate")
    ax.plot(table.index + 1, 100 * table["predicted"], "o-", color=GOLD, label="Mean predicted score")
    ax.axhline(100 * y.mean(), color=INK, ls="--", lw=0.9)
    ax.set(xlabel="Score decile (10 is the highest ranked)", ylabel="Percent", title="Implantation rate rises steadily with the model score")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "score_deciles.png"); plt.close(fig); written.append("score_deciles.png")

    # 6. Learning curves for every sweep run.
    fig, ax = plt.subplots(figsize=(7, 3.9))
    colours = [INK, GREY, INDIGO, ROSE, GOLD]
    for (name, history), colour in zip(report["histories"].items(), colours):
        ax.plot([h["epoch"] for h in history], [h["val_auc"] for h in history], "o-", ms=3.5, color=colour, label=_label(name),
                lw=2.2 if name == selected else 1.3)
    ax.set(xlabel="Epoch", ylabel="Validation ROC AUC", title="Sweep runs tracked in MLflow")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figure_dir / "learning_curves.png"); plt.close(fig); written.append("learning_curves.png")

    # 7. A single embryo timeline, to show what the model reads.
    typical = raw["t5"].between(49, 53) & (raw["s3"] > 7) & (raw["s2"] > 0.8) & raw["tSB"].between(92, 100)
    example = raw[(raw["transfer_day"] == 5) & raw[EVENTS].notna().all(axis=1) & typical].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.hlines(0, example["tPNf"] - 4, example["tEB"] + 4, color=INK, lw=1.2)
    for i, event in enumerate(EVENTS):
        ax.plot(example[event], 0, "o", color=INDIGO, ms=8)
        level = [0.5, -0.62, 1.25, -1.37][i % 4]
        ax.plot([example[event]] * 2, [0, level * 0.55], color=GREY, lw=0.7)
        ax.text(example[event], level, f"{event}\n{example[event]:.1f} h", ha="center", va="center", fontsize=8.5)
    ax.set(ylim=(-1.8, 1.7), yticks=[], xlabel="Hours post insemination", title="One embryo as the model sees it: twelve annotated events")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); fig.savefig(figure_dir / "embryo_timeline.png"); plt.close(fig); written.append("embryo_timeline.png")
    _ = EVENT_DESCRIPTIONS
    return written

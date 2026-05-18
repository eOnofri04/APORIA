"""
Reusable plotting helpers that appeared in multiple notebooks.

Only the genuinely-duplicated, configuration-independent plots live
here.  Notebook-specific figures (the bespoke layouts for Figures 1–7
of the paper) stay in their respective notebooks, since they each
contain visual tuning specific to one panel and aren't reused.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_boxplots_two_panels(
    results_lp: pd.DataFrame,
    model_names: dict[int, str],
    model_order: list[int],
    train_fraction: float,
    metrics: tuple[str, str] = ("accuracy", "f1"),
    ratio: tuple[float, float] = (6, 3),
    scale: float = 2,
    width_ratios: list[float] = [1, 1],
    xlims: list[tuple[float, float]] | None = None,
    cmap_name: str = "tab10",
):
    """Two-panel horizontal-boxplot view of a metric per model.

    For each model, the prompt-level distribution of ``metric_mean`` is
    drawn as a violin with a narrow box overlay.  Used for the
    accuracy/F1 panels in Section 4.2 (Figure 4) and for the saturation
    plots in the training-size sensitivity analysis.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=[scale * x for x in ratio],
        sharey=True,
        width_ratios=width_ratios,
    )

    cmap = plt.get_cmap(cmap_name)

    df_prompt = (
        results_lp
        .groupby(["model_id", "prompt_id", "train_fraction"])
        .agg(
            f1_mean=("f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            mean_n_train=("n_train", "mean"),
        )
        .reset_index()
    )

    for it, metric in enumerate(metrics):
        ax = axes[it]
        data: list = []
        labels: list = []
        colors: list = []

        for i, mid in enumerate(reversed(model_order)):
            name = model_names[mid]
            vals = df_prompt[
                (df_prompt["model_id"] == mid)
                & (df_prompt["train_fraction"] == train_fraction)
            ][f"{metric}_mean"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(name)
                colors.append(cmap(i % cmap.N))

        positions = np.arange(1, len(data) + 1)

        # violins
        vp = ax.violinplot(
            data,
            widths=0.75,
            positions=positions,
            vert=False,
            showmeans=False,
            showextrema=False,
        )
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.35)
            body.set_edgecolor("none")

        # boxes
        bp = ax.boxplot(
            data,
            positions=positions,
            vert=False,
            widths=0.25,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c)
            box.set_alpha(0.6)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_title(metric.capitalize())
        ax.grid(True, axis="x", alpha=0.4)
        if xlims is not None:
            ax.set_xlim(xlims[it])

    fig.tight_layout()
    return fig, axes

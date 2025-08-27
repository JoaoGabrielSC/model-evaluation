import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List


def plot_metric_histograms(results: List[dict], output_dir: str) -> None:
    metrics = ["accuracy", "f1_score", "precision"]
    models = sorted(set(r["model"] for r in results))
    distances = sorted(set(r["dist"] for r in results))

    data = {metric: np.zeros((len(models), len(distances))) for metric in metrics}

    for r in results:
        m_idx = models.index(r["model"])
        d_idx = distances.index(r["dist"])
        for metric in metrics:
            data[metric][m_idx, d_idx] = r.get(metric, 0)

    n_metrics = len(metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
    )
    width = 0.8 / len(distances)

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for j, dist in enumerate(distances):

            positions = np.arange(len(models)) + j * width
            values = data[metric][:, j]
            ax.bar(positions, values, width=width, label=dist)

        ax.set_xticks(np.arange(len(models)) + width * (len(distances) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} por Modelo/Distância")
        ax.legend(title="Distância")

        for j, dist in enumerate(distances):
            positions = np.arange(len(models)) + j * width
            values = data[metric][:, j]
            for x, val in zip(positions, values):
                ax.text(x, val * 0.95, f"{val:.2f}", ha="center", fontsize=8)

    for i in range(n_metrics, nrows * ncols):
        fig.delaxes(axes[i // ncols][i % ncols])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_mosaic.png"))
    plt.close()

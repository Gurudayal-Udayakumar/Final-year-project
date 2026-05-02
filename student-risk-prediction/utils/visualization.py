"""Reusable plotting helpers for Streamlit pages."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_correlation_heatmap(df: pd.DataFrame):
    """Create a readable upper-triangle heatmap for numeric correlations."""
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(numeric_only=True)

    renamed = corr.copy()
    renamed.columns = [column.replace("_", " ").title() for column in renamed.columns]
    renamed.index = [index.replace("_", " ").title() for index in renamed.index]
    mask = np.tril(np.ones_like(renamed, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        renamed,
        mask=mask,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        linewidths=0.5,
        square=False,
        cbar_kws={"shrink": 0.8, "label": "Correlation strength"},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap for Numeric Student Features", fontsize=14, pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    return fig

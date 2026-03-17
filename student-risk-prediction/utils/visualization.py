"""Reusable plotting helpers for Streamlit pages."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def build_correlation_heatmap(df: pd.DataFrame):
    """Create a seaborn heatmap figure for numeric correlations."""
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig

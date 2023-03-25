import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_results(axs, epochs, results, title):
    axs.plot(np.arange(1, epochs + 1), results["train"], label="Train")
    axs.plot(np.arange(1, epochs + 1), results["val"], label="Val")
    axs.set_xticks(np.arange(1, epochs + 1, step=2))
    axs.set_title(title)
    axs.legend()


def plot_correlated_features(feat_1, feat_2, n_cols=3, sx=4, sy=5):
    # Plot w.r.t feat_1
    n_rows = math.ceil(feat_1.nunique() / n_cols)
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(sx * n_cols, sy * n_rows)
    )
    faxs = axs.ravel()

    for i, v in enumerate(feat_1.drop_duplicates().sort_values()):
        idx = feat_1[feat_1 == v].index
        feat_2_sub = feat_2[idx].sort_values()
        faxs[i].plot(range(len(idx)), feat_2_sub)

        faxs[i].set_title(f"{feat_1.name}=={v}")
        faxs[i].title.set_fontsize(9)

        y_min, y_max, y_step = feat_2_sub.min(), feat_2_sub.max() + 1, 6
        faxs[i].set_yticks(
            np.arange(y_min, y_max, step=math.ceil((y_max - y_min) / y_step))
        )
        b_min, b_max, step = 0, len(idx), 6
        faxs[i].set_xticks(
            np.arange(b_min, b_max, step=math.ceil((b_max - b_min) / step))
        )

    for i in range(1, 1 + (n_rows * n_cols - feat_1.nunique())):
        faxs[-i].set_visible(False)
    fig.show()
    

def plot_dataframe(data, labels=None, vmin=-9, vmax=0.15, figsize=None, s=4):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect="auto", cmap="RdBu", vmin=vmin, vmax=vmax)
    plt.yticks(range(0, data.shape[1]))
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = -0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(
            labels.index,
            np.ones(len(labels)) * lvl,
            s=s,
            color=plt.get_cmap("tab10")(np.mod(labels, 10)),
        )
    plt.tight_layout()


def plot_df_histograms(df):
    assert isinstance(df, pd.DataFrame)

    n_cols = 3
    n_rows = math.ceil(df.shape[1] / n_cols)
    x_size, y_size = 3, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_size * n_cols, y_size * n_rows))
    axes = axes.ravel()

    for i, col in enumerate(df.columns):
        # bins = np.linspace(df[col].min(), df[col].max(), 20)
        bins = 4
        df[col].plot(kind="hist", bins=bins, ax=axes[i], title=col, edgecolor='black')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        # xticks = np.arange(df[col].min(), df[col].max()+1, 1) # adjust xticks
        axes[i].set_xticks(xticks)

    fig.tight_layout()
    fig.show()



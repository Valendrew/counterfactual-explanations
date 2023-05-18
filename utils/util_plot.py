import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_results(axs, epochs, results, title):
    axs.plot(np.arange(1, epochs + 1), results["train"], label="Train")
    axs.plot(np.arange(1, epochs + 1), results["val"], label="Val")
    axs.set_xticks(np.arange(1, epochs + 1, step=4))
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
    plt.show()


def plot_compare_models(X, y, y_pred, names, title):
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        plt.plot(X, y_pred[i], color=f"C{i+1}", label=name)

    plt.scatter(X, y, color="C0", label="Data")

    plt.title(title)
    plt.legend()
    plt.show()


def plot_series_distribution(series, title, boxplot=False):
    n_rows = 2
    fig, ax = plt.subplots(figsize=(8, 4*n_rows), nrows=n_rows)
    ax = ax.ravel()

    # histogram of conversion rates
    series.hist(bins=20, edgecolor="black", linewidth=1.2, ax=ax[0])
    ax[0].set_xticks(np.arange(round(series.min()), round(series.max()) + 2, 3))
    # boxplot of conversion rates
    if boxplot:
        series.plot.box(whis=1.5, ax=ax[1])
        # get median conversion rate
        median_conv_rate = series.median()
        # yticks lower than median
        yticks_lower = np.arange(round(series.min()), round(median_conv_rate), 4)
        # yticks higher than median
        yticks_higher = np.arange(round(median_conv_rate), round(series.max()) + 1, 4)
        # set xticks
        ax[1].set_yticks(np.concatenate((yticks_lower, yticks_higher)))
    else:
        ax[1].set_visible(False)

    fig.suptitle("Distribution of conversion rates")
    fig.tight_layout()

def plot_relative_changes(mean_changed: pd.Series, axs, title: str, remove_right):
    mean_changed.plot.barh(ax=axs, title=title, rot=0)
    # set title fontsize
    axs.title.set_fontsize(12)

    # negative values in red
    neg_idx = mean_changed[mean_changed <= 0].index
    for i in neg_idx:
        idx = mean_changed.index.get_loc(i)
        axs.get_children()[idx].set_color("r")

    # add vertical line at 0
    axs.axvline(x=0,linewidth= 1, color='k')

    # remove axis
    if remove_right:
        axs.spines['right'].set_visible(False)
    for s in ['top', 'bottom', 'left']:
        axs.spines[s].set_visible(False)

    # remove values on x-axis
    axs.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # remove values on y-axis
    axs.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)

    # add values directly on bars
    for i, v in enumerate(mean_changed):
        x_pos = 0.5 if v < 0 else v + 0.5
        axs.text(x_pos, i - 0.1, f"{v:.1f}%", color='black', fontweight='bold', fontsize=10)
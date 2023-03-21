import matplotlib.pyplot as plt
import numpy as np


def plot_model_results(axs, epochs, results, title):
    axs.plot(np.arange(1, epochs + 1), results["train"], label="Train")
    axs.plot(np.arange(1, epochs + 1), results["val"], label="Val")
    axs.set_xticks(np.arange(1, epochs + 1, step=2))
    axs.set_title(title)
    axs.legend()

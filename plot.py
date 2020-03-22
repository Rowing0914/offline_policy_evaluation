import numpy as np
import matplotlib.pyplot as plt

from data.data_manager import names


def plot_bar_chart(df, plot_name="bias", fig_name="bias.png"):
    """ Plotting function, based on the plots on the paper

    :param df: a dataframe which contains resulting values w.r.t a certain metric
    :param plot_name: a name of plot
    :param fig_name: a name of the file to save the resulting figure
    """
    x = np.arange(len(names))  # the label locations
    width = 0.2  # the width of the bars
    margin = 0.05

    fig, ax = plt.subplots()
    ax.bar(x - (width + margin), ips_est, width, color="r", label="IPS")
    ax.bar(x, dr_est, width, color="g", label="DR")
    ax.bar(x + (width + margin), dm_est, width, color="b", label="DM")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(plot_name)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig(fig_name)


if __name__ == '__main__':
    ips_est = np.random.rand(len(names))
    dm_est = np.random.rand(len(names))
    dr_est = np.random.rand(len(names))
    plot_bar_chart(ips_est, dm_est, dr_est)
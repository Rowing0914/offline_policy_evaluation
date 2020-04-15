import numpy as np
import matplotlib.pyplot as plt


def plot_bar_chart(df, plot_name="bias", fig_name="bias.png"):
    """ Plotting function, based on the plots on the paper

    :param df: a dataframe which contains resulting values w.r.t a certain metric
    :param plot_name: a name of plot
    :param fig_name: a name of the file to save the resulting figure
    """

    fig = plt.figure()
    ax = plt.subplot(111)

    est_names = df["algo"].values
    del df["algo"]

    for col in list(df.columns):
        ax.plot(df[col].values, label=col)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel(plot_name)
    ax.set_xlabel("Estimators")
    ax.set_xticks(np.arange(len(est_names)))
    ax.set_xticklabels(labels=est_names, rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_name, bbox_extra_artists=(legend,), bbox_inches = 'tight')
    plt.clf()


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("bias.csv")
    plot_bar_chart(df=df, plot_name="bias", fig_name="bias.png")
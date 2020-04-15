import numpy as np
import matplotlib.pyplot as plt


def plot_bar_chart(df, plot_name="bias", fig_name="bias.png"):
    """ Plotting function, based on the plots on the paper

    :param df: a dataframe which contains resulting values w.r.t a certain metric
    :param plot_name: a name of plot
    :param fig_name: a name of the file to save the resulting figure
    """

    est_names = df["algo"].values
    del df["algo"]

    for col in list(df.columns):
        plt.plot(df[col].values, label=col)

    plt.ylabel("Bias")
    plt.xlabel("Estimators")
    plt.xticks(ticks=np.arange(len(est_names)), labels=est_names, rotation=90)
    plt.grid()
    plt.legend()
    plt.title(plot_name)
    plt.savefig(fig_name)
    plt.clf()


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("./test/bias.csv")
    plot_bar_chart(df=df, plot_name="bias", fig_name="bias.png")
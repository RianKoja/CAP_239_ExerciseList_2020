# Standard imports:
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster


def k_means_graph(ticker1, ticker2, df, k):
    # based on https://www.springboard.com/blog/data-mining-python-tutorial/
    df_reduced = df[[ticker1, ticker2]].dropna()
    np_array = np.array(df_reduced)
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(np_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Create new figure:
    fig = plt.gcf()
    if fig.get_axes():
        plt.figure(plt.gcf().number+1)
    fig = plt.gcf()
    fig.set_size_inches(13, 8)

    for i in range(k):
        # select only data observations with cluster label == i
        ds = np_array[np.where(labels == i)]
        # plot the data observations
        plt.plot(ds[:, 0], ds[:, 1], 'o', markersize=7)
        # plot the centroids
        lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
        # make the centroid x's bigger
        plt.setp(lines, ms=15.0)
        plt.setp(lines, mew=4.0)

    ax = plt.gca()

    # Added tickers as labels:
    this_df = df.dropna(subset=[ticker1, ticker2])
    x = this_df[ticker1].tolist()
    y = this_df[ticker2].tolist()
    for ii, txt in enumerate(this_df['Código do fundo'].tolist()):
        ax.annotate(txt, (x[ii], y[ii]), fontsize=7)

    plt.title("K-Means grouping for " + ticker1 + " and " + ticker2 + " with k=" + str(k))
    plt.xlabel(ticker1)
    plt.ylabel(ticker2)
    ax.grid(axis='both')
    plt.tight_layout()
    plt.draw()

    return kmeans.inertia_


def save_all(xlsx_name):
    for ii in range(1, plt.gcf().number+1):
        plt.figure(num=ii)
        plt.savefig("fig_{:02d}".format(ii) + xlsx_name + ".png")
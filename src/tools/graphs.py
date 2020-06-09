# Standard imports:
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn import cluster


def plot_k_means(df, parameters, doc, full_name):
    # Pick K for K-Means:
    optimal_k = pick_k(df, parameters)
    # Build final K-Means object:
    ex_kmeans = k_means(df, optimal_k, parameters)
    # Add kmeans grouping to data frame:
    df['kmeans'] = ex_kmeans.labels_
    # Plot the k-means grouping
    sns_plot = sns.pairplot(df, hue="kmeans", vars=parameters, height=1.3)
    plt.tight_layout(pad=2)
    plt.suptitle("K-Means Grouping for " + full_name + "\non space " + ", ".join(parameters), fontsize=10)
    doc.add_fig()
    plt.draw()
    return ex_kmeans


def pick_k(df, parameters):
    silhouette_scores = []
    k_range = list(range(2, 10))
    for ii in k_range:
        kmean_obj = k_means(df, ii, parameters)
        if ii != 1:
            silhouette_scores.append(silhouette_score(df[parameters], kmean_obj.labels_, metric='euclidean'))
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k


def k_means(df, k, parameters):
    df_reduced = df[parameters]
    np_array = np.array(df_reduced)
    kmeans = cluster.KMeans(n_clusters=k)
    with warnings.catch_warnings():  # suppress the warning for the case perfect grouping is found before end of loop
        warnings.filterwarnings("ignore")
        kmeans.fit(np_array)
    #  labels = kmeans.labels_
    #  centroids = kmeans.cluster_centers_
    return kmeans


def save_all(xlsx_name):
    for ii in range(1, plt.gcf().number+1):
        plt.figure(num=ii)
        plt.savefig("fig_{:02d}".format(ii) + xlsx_name + ".png")
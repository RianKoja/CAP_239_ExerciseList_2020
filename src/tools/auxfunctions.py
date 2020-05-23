########################################################################################################################
# Auxiliary functions to perform  CAP 239 exercise lists, exercises 1 to 3. Should be refactored for better
# architecture.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################
# Standard imports:

import pandas as pd
from pandas.compat import BytesIO
import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tools import createdocument

# local imports:


# This function stacks several one-line data frames ot create a larger set of desired size.
def generatedataframe(algorithm):
    df = pd.DataFrame(columns=['mean', 'variance', 'skewness', 'kurtosis', 'Type'])
    dataframe = algorithm.makedataframe(df)
    print("DATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAME")
    print(dataframe)
    print("DATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAMEDATAFRAME")
    return dataframe


def k_means(df, k):
    df_reduced = df[['variance', 'skewness', 'kurtosis']]
    np_array = np.array(df_reduced)
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(np_array)
    #  labels = kmeans.labels_
    #  centroids = kmeans.cluster_centers_
    return kmeans


def plot_inertias(df, algorithm, doc_report):
    inertias = []
    silhouette_scores = []
    for ii in range(1, 10):
        kmean_obj = k_means(df[['variance', 'skewness', 'kurtosis']], ii)
        inertias.append(kmean_obj.inertia_)
        if ii != 1:
            silhouette_scores.append(silhouette_score(df[['variance', 'skewness', 'kurtosis']], kmean_obj.labels_,
                                                      metric='euclidean'))

    plt.figure()
    plt.plot(range(1, 10), inertias)
    plt.plot(range(1, 10), inertias, 'bo')
    plt.ylabel("Inertia for K-Means Clustering")
    plt.xlabel("Number of Clusters")
    plt.title("Inertia for K-Means Clustering , data generated with " + algorithm.name)
    plt.grid(which='both', axis='both')
    plt.draw()
    memfile = BytesIO()
    plt.savefig(memfile)
    doc_report.add_fig(memfile)

    # Also plot silhouette score:
    plt.figure()
    plt.plot(range(2, 10), silhouette_scores)
    plt.plot(range(2, 10), silhouette_scores, 'bo')
    plt.ylabel("Silhouette Scores for K-Means Clustering")
    plt.xlabel("Number of Clusters")
    plt.title("Silhouette Scores for K-Means Clustering, data generated with " + algorithm.name)
    plt.grid(which='both', axis='both')
    plt.draw()
    memfile = BytesIO()
    plt.savefig(memfile)
    doc_report.add_fig(memfile)

    print(silhouette_scores)
    print("max(silhouette_scores) = " + str(max(silhouette_scores)))
    optimal_k = 2 + silhouette_scores.index(max(silhouette_scores))
    print("For data generated with " + algorithm.name + " the maximum silhouette occurs for " + str(optimal_k) + " clusters")
    return optimal_k


def exercises_1_3(algorithm, doc_report):
    print("This is ", __name__, " in ", __file__)

    ex1_df = generatedataframe(algorithm)

    # run K-means elbow and silhouette plot:
    selected_k = plot_inertias(ex1_df, algorithm, doc_report)

    ex_kmeans = k_means(ex1_df, selected_k)

    # Add kmeans grouping to data frame:
    ex1_df['kmeans'] = ex_kmeans.labels_

    # Save in excel for logging/debbuging:
    ex1_df.to_excel(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount', algorithm.name + '_auxfunctions_test.xlsx'))

    # plot heatmap with incidence versus case:
    # Create heatmap:
    plt.figure()
    cross_tab = pd.crosstab(ex1_df['Type'], ex1_df['kmeans'])
    print(cross_tab)
    sns.set()
    sns_plot = sns.heatmap(cross_tab, cmap="YlGnBu", annot=True, cbar=False, fmt="d", square=True, linewidths=0.5)
    plt.title("Incidence Matrix obtained from k-means based on \n" + 'N_elements' + " and " + 'kmeans' + " with k=" +
              str(selected_k))
    memfile = BytesIO()
    plt.savefig(memfile)
    doc_report.add_fig(memfile)

    # Plot the k-means grouping
    plt.figure()
    sns_plot = sns.pairplot(ex1_df, hue="kmeans", vars=['variance', 'skewness', 'kurtosis'])
    memfile2 = BytesIO()
    plt.savefig(memfile2)
    doc_report.add_fig(memfile2)

    # For comparison, color mark the number of elements
    sns_plot = sns.pairplot(ex1_df, hue='Type', vars=['variance', 'skewness', 'kurtosis'])
    memfile3 = BytesIO()
    plt.savefig(memfile3)
    doc_report.add_fig(memfile3)


# This module is meant to provide functions imported elsewhere, nonetheless, it is useful to run it directly for
# testing and development purposes
if __name__ == '__main__':
    # Use fixed seed, so results don't change between runs of the same algorithm:
    np.random.seed(82745949)

    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()

    # For questions 1-3, do the same procedure importing a different signal generator:
    from generators import GRNG

    a_algorithm = GRNG()
    # Run the function thar performs exercises:
    exercises_1_3(a_algorithm, test_report)

    test_report.finish()

    plt.show()

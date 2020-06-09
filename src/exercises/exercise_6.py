
# Standard imports:
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn import cluster

# Local imports:
from tools import specplus, stat, createdocument
from generators import grng, colorednoise, pmodel, logis, henon


def plot_beta_compare(df, full_name, doc):
    df.plot.scatter(x='beta', y='beta_theoretical')
    # Compute person correlation coefficient:
    correlation = df[['beta', 'beta_theoretical']].corr()

    # Format chart:
    plt.grid('both')
    plt.xlabel(r'$\beta$ from linear regression.')
    plt.ylabel(r'$\beta$ from detrended fluctuation analysis $\beta = 2\alpha - 1$')
    plt.title(r'Assessment of $\beta$ computation for ' + full_name + "\nCorrelation = %2.2f" % correlation.iloc[0, 1])
    plt.draw()
    doc.add_fig()


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


def pick_k(df, parameters):
    silhouette_scores = []
    for ii in range(1, 10):
        kmean_obj = k_means(df, ii, parameters)
        if ii != 1:
            silhouette_scores.append(silhouette_score(df[parameters], kmean_obj.labels_, metric='euclidean'))

    optimal_k = 2 + silhouette_scores.index(max(silhouette_scores))
    return optimal_k


def plot_k_means(df, parameters, doc, full_name):
    # Pick K for K-Means:
    optimal_k = pick_k(df, parameters)
    # Build final K-Means object:
    ex_kmeans = k_means(df, optimal_k, parameters)
    # Add kmeans grouping to data frame:
    df['kmeans'] = ex_kmeans.labels_
    # Plot the k-means grouping
    sns_plot = sns.pairplot(df, hue="kmeans", vars=parameters, height=1.2)
    plt.tight_layout(pad=2)
    plt.suptitle("K-Means Grouping for " + full_name, fontsize=10)
    doc.add_fig()
    plt.draw()


def run(doc):
    names = ('GNRG', 'Color', 'P_model_025_exogen_beta04', 'logistic_rho3.88_tau1.1', 'henon_a1.38_b0.22')
    functions = (lambda: grng.time_series(2 ** np.random.randint(6, 13), 1),
                 lambda: colorednoise.powerlaw_psd_gaussian(np.random.uniform(0, 2), 8192),
                 lambda: pmodel.pmodel(8192, np.random.uniform(0.18, 0.42), 0.4)[1],
                 lambda: logis.logistic_series(np.random.uniform(3.85, 3.95), 0.5, 8192)[1],
                 lambda: henon.henon_series(np.random.uniform(1.35, 1.42), np.random.uniform(0.21, 0.31), 8192)[1])

    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator', 'P-Model', 'Logistic Map', 'Henon Map')
    sizes = (160, 120, 120, 120)
    #sizes = (1, 1, 1, 1)
    columns = ['skewness', 'skewness²', 'kurtosis', 'alpha', 'beta', 'beta_theoretical']
    for (name, func, full_name, size) in zip(names, functions, full_names, sizes):
        df = pd.DataFrame(columns=columns)
        for ii in range(0, size):
            # Generate a time series:
            data = func()
            # Compute parameters:
            ds = stat.series2datasetline(data)
            alpha, beta_t, beta = specplus.main(data)
            plt.close()
            # Append to data frame:
            df = df.append(pd.DataFrame([[ds['skewness'], ds['skewness']**2, ds['kurtosis'], alpha, beta, beta_t]],
                           columns=columns), ignore_index=True, sort=False)
        plot_beta_compare(df, full_name, doc)
        for parameter_set in (['skewness²', 'kurtosis', 'beta'], ['skewness²', 'kurtosis', 'alpha']):

            # Plot K-Means:
            plot_k_means(df, parameter_set, doc, full_name)


# Sample execution
if __name__ == '__main__':
    np.random.seed(82745949)
    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report)
    test_report.finish()
    plt.show()
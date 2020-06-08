
# Standard imports:
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn import cluster

# Local imports:
from tools import specplus, stat
from generators import grng, colorednoise, pmodel, logis, henon


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
    inertias = []
    silhouette_scores = []
    for ii in range(1, 10):
        kmean_obj = k_means(df[parameters], ii)
        inertias.append(kmean_obj.inertia_)
        if ii != 1:
            silhouette_scores.append(silhouette_score(df[parameters], kmean_obj.labels_, metric='euclidean'))


def plot_s_k_b(df):

    # Plot the k-means grouping
    plt.figure()
    sns_plot = sns.pairplot(df, hue="kmeans", vars=['variance', 'skewness', 'kurtosis'])
    pass


def run():
    # Henon map not used as it causes issues with the waipy module.
    names = ('GNRG', 'Color') #, 'P_model_025_exogen_beta04', 'logistic_rho3.88_tau1.1')  # 'henon_a1.38_b0.22',
    functions = (lambda: grng.time_series(8192, 1),
                 lambda: colorednoise.powerlaw_psd_gaussian(1, 8192)) #,
                  #lambda: pmodel.pmodel(8192, 0.25, 0.4)[1],
                  #lambda: logis.logistic_series(3.88, 1.1, 8192)[1])

    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator') #, 'P-Model', 'Logistic Map')  # 'Henon Map'
    sizes = (80, 60, 60, 60)
    columns = ['skewness', 'skewness_square', 'kurtosis', 'alpha', 'beta', 'beta_theoretical']
    for (name, func, full_name, size) in zip(names, functions, full_names, sizes):
        skews, kurts, betas, alphas = ([] for i in range(0, 4))
        df = pd.DataFrame(columns=columns)
        for ii in range(0, size):
            # Generate a time series:
            data = func()
            # Compute parameters:
            ds = stat.series2datasetline(data)
            alpha, beta_t, beta = specplus.main(data)
            plt.show()
            plt.close('all')
            # Append to data frame:
            df = df.append(pd.DataFrame([[ds['skewness'], ds['skewness']**2, ds['kurtosis'], alpha, beta, beta_t]],
                           columns=columns), ignore_index=True, sort=False)

        print(df)


# Sample execution
if __name__ == '__main__':
    run()

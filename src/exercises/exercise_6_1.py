
# Standard imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports:
from tools import createdocument, graphs, specplus, stat
from generators import grng, colorednoise, pmodel, logis, henon


def plot_beta_compare(df, full_name, doc):
    df.plot.scatter(x=r'$\beta$', y='beta_theoretical')
    # Compute person correlation coefficient:
    correlation = df[[r'$\beta$', 'beta_theoretical']].corr()

    # Format chart:
    plt.grid('both')
    plt.xlabel(r'$\beta$ from linear regression.')
    plt.ylabel(r'$\beta$ from detrended fluctuation analysis $\beta = 2\alpha - 1$')
    plt.title(r'Assessment of $\beta$ computation for ' + full_name + "\nCorrelation = %2.2f" % correlation.iloc[0, 1])
    plt.draw()
    doc.add_fig()


def run(doc):
    np.random.seed(82745949)  # Use here to avoid discrepant results.
    doc.add_heading('Exercise 6.1', level=2)
    doc.add_paragraph("The P-Model will be excluded, since it provides data series with high kurtosis and skewness" +
                      "that would prevent effective clustering with K-Means")
    names = ('GNRG', 'Color', #'P_model_025_exogen_beta04',
              'logistic_rho3.88_tau1.1', 'henon_a1.38_b0.22')
    functions = (lambda: grng.time_series(2 ** np.random.randint(6, 13), 1),
                 lambda: colorednoise.powerlaw_psd_gaussian(np.random.uniform(0, 2), 8192),  # lambda: pmodel.pmodel(8192, np.random.uniform(0.18, 0.42), 0.4)[1],
                 lambda: logis.logistic_series(np.random.uniform(3.85, 3.95), 0.5, 8192)[1],
                 lambda: henon.henon_series(np.random.uniform(1.35, 1.41), np.random.uniform(0.21, 0.31), 8192)[1])

    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator', #'P-Model',
                  'Logistic Map', 'Henon Map')
    sizes = (80, 60, 60, 60)
    columns = ['skewness', 'skewness²', 'kurtosis', r'$\alpha$', r'$\beta$', 'beta_theoretical']
    parameter_spaces = (['skewness²', 'kurtosis', r'$\beta$'], ['skewness²', 'kurtosis', r'$\alpha$'])
    df_global = pd.DataFrame(columns=columns)

    for (name, func, full_name, size) in zip(names, functions, full_names, sizes):
        doc.document.add_heading("For " + full_name + ":", level=3)
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
        # Append to global data frame:
        df['Generator'] = full_name
        df_global = df_global.append(df, ignore_index=True, sort=False)
        # Check beta relation:
        plot_beta_compare(df, full_name, doc)

        # Plot K-Means for both sets of parameters:
        for parameter_set in parameter_spaces:
            graphs.plot_k_means(df, parameter_set, doc, full_name)

    # Now run the analysis for the global data frame:
    doc.add_heading("Combining all data series:", level=3)
    # Check beta relation:
    plot_beta_compare(df_global, "All series", doc)
    # Group with K-Means
    k_means_out = [graphs.plot_k_means(df_global, parameter_set, doc, "All series")
                   for parameter_set in parameter_spaces]
    # Create incidence matrix:
    for k_means_obj, parameter_set in zip(k_means_out, parameter_spaces):
        plt.figure()
        cross_tab = pd.crosstab(df_global['Generator'], k_means_obj.labels_)
        sns.heatmap(cross_tab, cmap="YlGnBu", annot=True, cbar=False, fmt="d", square=True, linewidths=0.5)
        plt.title("Incidence Matrix obtained from k-means based on " + " x ".join(parameter_set))
        doc.add_fig()
    # Return this for usage in exercise 6.2:
    return k_means_out


# Sample execution
if __name__ == '__main__':
    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report)
    test_report.finish()
    plt.show()

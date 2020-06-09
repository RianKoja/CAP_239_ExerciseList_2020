
# Standard imports:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# Local imports:
from generators import grng, colorednoise, pmodel, logis, henon
from tools import createdocument, getdata, graphs
import tools.mfdfa_ss as mfdfa


def run(doc):
    doc.add_heading("Exercise 7.1")
    doc.add_paragraph("The MDFDA files have been fully refactored for the purpose of this work, with many additional" +
                      " parameters being computed and shown in charts, including" + r'$\Psi$')

    doc.add_heading("Exercise 7.2")
    doc.add_paragraph("For each signal generator, only one singularity spectrum will be plotted, but the aggregated " +
                      " statistics will be shown for the next item.")

    # Control the random seed so results are consistent between runs:
    np.random.seed(182745949)

    # Prepare iterations:
    names = ('GNRG', 'Color', 'P_model_025_exogen_beta04', 'logistic_rho3.88_tau1.1', 'henon_a1.38_b0.22')
    functions = (lambda: grng.time_series(2 ** np.random.randint(6, 13), 1),
                 lambda: colorednoise.powerlaw_psd_gaussian(np.random.uniform(0, 2), 8192),
                 lambda: pmodel.pmodel(8192, np.random.uniform(0.18, 0.42), 0.4)[1],
                 lambda: logis.logistic_series(np.random.uniform(3.85, 3.95), 0.5, 8192)[1],
                 lambda: henon.henon_series(np.random.uniform(1.35, 1.4), np.random.uniform(0.21, 0.31), 8192)[1])

    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator', 'P-Model', 'Logistic Map', 'Henon Map')
    sizes = (80, 60, 60, 60, 60)
    columns = ['skewness²', r'$\Psi$', 'generator']
    df_all = pd.DataFrame(columns=columns)
    for (name, func, full_name, size) in zip(names, functions, full_names, sizes):
        doc.add_heading(full_name, level=3)
        df_tmp = pd.DataFrame(columns=columns)
        for ii in range(0, size):
            # Generate a time series:
            data = func()
            plt.close('all')
            ret = mfdfa.main(data)
            df_tmp = df_tmp.append(pd.DataFrame([[skew(data)**2, ret['Psi'], full_name]], columns=columns))
        doc.add_fig()
        # Add K-means chart:
        graphs.plot_k_means(df_tmp, ['skewness²', r'$\Psi$'], doc, full_name)

        df_all = df_all.append(df_tmp, ignore_index=True, sort=False)
    doc.add_heading("For all generators:", level=3)
    graphs.plot_k_means(df_all, ['skewness²', r'$\Psi$'], doc, "All series")

    # Next section:
    doc.add_heading("Exercise 7.3", level=2)
    doc.add_paragraph("")
    # Load country data:
    df_all = getdata.acquire_data('all')
    country_list = list(set(df_all['location'].to_list()))
    columns = ['skewness²', r'$\Psi$', 'Country']
    df = pd.DataFrame(columns=columns)
    for location in country_list:
        df_owd = df_all[df_all['location'] == location]
        data = df_owd['new_cases'].to_list()
        if len(list(set(data))) > 11:
            ret = mfdfa.main(data)
            plt.close('all')
            df = df.append(pd.DataFrame([[skew(data) ** 2, ret['Psi'], location]], columns=columns))

    kmeans_obj = graphs.plot_k_means(df, ['skewness²', r'$\Psi$'], doc, "New Cases of COVID-19 By Country")
    df['group'] = kmeans_obj.labels_
    doc.add_paragraph("Now, we print the results for each country along with the grouping proposed:")
    doc.add_paragraph(df.to_string())


# Sample execution:
if __name__ == '__main__':
    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report)
    test_report.finish()
    print("Finished ", __file__)
    plt.show()

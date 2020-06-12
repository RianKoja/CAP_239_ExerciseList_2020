
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
    doc.add_heading("Exercise 7.1", level=2)
    doc.add_heading("Exercise 7.1", level=3)
    doc.add_paragraph("The MDFDA files have been fully refactored for the purpose of this work, with many additional" +
                      " parameters being computed and shown in charts, including" + r'$\Psi$')

    doc.add_heading("Exercise 7.2", level=3)
    doc.add_paragraph("For each signal generator, only one singularity spectrum will be plotted, but the aggregated " +
                      " statistics will be shown for the next item.")

    # Control the random seed so results are consistent between runs:
    np.random.seed(182745949)

    # Prepare iterations:
    names = ('GNRG', 'Color', 'P_model_025_exogen_beta04', 'logistic_rho3.88_tau1.1', 'henon_a1.38_b0.22')
    functions = (lambda: grng.time_series(2 ** np.random.randint(6, 13), 1),
                 lambda: colorednoise.powerlaw_psd_gaussian(np.random.uniform(0, 2), 8192),
                 lambda: pmodel.pmodel(n_values=8192, p=np.random.uniform(0.18, 0.42), slope=0.4)[1],
                 lambda: logis.logistic_series(np.random.uniform(3.85, 3.95), 0.5, 8192)[1],
                 lambda: henon.henon_series(np.random.uniform(1.35, 1.4), np.random.uniform(0.21, 0.31), 8192)[1])

    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator', 'P-Model', 'Logistic Map', 'Henon Map')
    sizes = (80, 60, 60, 60, 60)
    columns = ['skewness²', r'$\Psi$', 'generator']
    df_all = pd.DataFrame(columns=columns)
    xd = list()
    yd = list()
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
        # Save data form the last chart:
        line = plt.gca().get_lines()[0]
        xd.append(line.get_xdata())
        yd.append(line.get_ydata())
        # Add K-means chart:
        graphs.plot_k_means(df_tmp, ['skewness²', r'$\Psi$'], doc, full_name)

        df_all = df_all.append(df_tmp, ignore_index=True, sort=False)
    doc.add_heading("For all generators:", level=3)
    graphs.plot_k_means(df_all, ['skewness²', r'$\Psi$'], doc, "All series")

    plt.figure()
    for x, y, full_name in zip(xd, yd, full_names):
        plt.plot(x, y, 'o-', label=full_name)
    plt.title("Comparing Singularity Spectra")
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\alpha)$')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.grid('on', which='both')
    plt.tight_layout()
    doc.add_heading("Comparing singularity Spectra", level=4)
    doc.add_paragraph("Here we show a comparative example of the singularity spectrum for time series generated from " +
                      "different methods, in particular, we see the much wider spectrum of the series generated with " +
                      "P-Model, which is also quite symmetrical, while for the other models, the spectrum is " +
                      "left-truncated, which indicates that the spectrum is insensitive to larger local fluctuations.")
    doc.add_paragraph("The spectrum of the P-Model being the widest, also indicates a higher degree of " +
                      "multifractality and data complexity.")
    doc.add_fig()
    # Next section:
    doc.add_heading("Exercise 7.3", level=3)
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

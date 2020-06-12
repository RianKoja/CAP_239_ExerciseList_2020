
# Standard imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports:
from generators import pmodel
from tools import createdocument, getdata, soc


def addplot(data, label, ymin=None):
    np.seterr(divide='ignore')
    prob_gamma, counts = soc.soc_main(data)
    log_prob = np.log10(prob_gamma)
    p = np.array(prob_gamma)
    p = p[np.nonzero(p)]
    c = counts[np.nonzero(counts)]
    log_p = np.log10(p)
    a = (log_p[np.argmax(c)] - log_p[np.argmin(c)]) / (np.max(c) - np.min(c))
    b = log_prob[0]
    y = b * np.power(10, (a * counts))
    x = np.log10(counts)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if ymin is None:
        plt.plot(x, y, marker=".", label=label)
    elif True in (yt < ymin for yt in y):
        plt.plot(x, y, marker=".", label=label)
    np.seterr(divide='warn')


def run(doc):
    doc.add_heading("Exercise 9", level=2)
    doc.add_heading("Exercise 9.1", level=3)
    doc.add_paragraph("Due to the low diversity of numbers contained in a p-model generated time series, the " +
                      "aggregation used by the soc.py algorithm often produces few points or empty bins, resulting in " +
                      "poor charts for this generator.")

    # Control the random seed so results are consistent between runs:
    np.random.seed(1827459459)

    for name, p in zip(('Endogenous', 'Exogenous'), ([0.32, 0.42], [0.18, 0.28])):
        doc.add_heading("For endogenous series:", level=2)
        plt.figure()
        for ii in range(0, 50):
            data = pmodel.pmodel(n_values=8192, p=np.random.uniform(p[0], p[1]), slope=np.random.randint(1,3))[1]
            addplot(data, str(ii))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
        plt.grid('both')
        plt.xlabel('log(ni)')
        plt.ylabel('log(Yi)')
        plt.title("SOC for " + name + " Series")
        plt.tight_layout()
        plt.draw()
        doc.add_fig()

    # Now for 9.2:
    doc.add_heading("Exercise 9.2", level=3)
    # Get list of countries:
    df_all = getdata.acquire_data('all')
    country_list = list(set(df_all['location'].to_list()))
    column_list = ['skewness²', 'kurtosis', r'$\alpha$', r'$\beta$', r'$\beta_t$', 'name']
    excluded = list()
    plt.figure()
    for location in country_list:
        df_owd = df_all[df_all['location'] == location]
        data = df_owd['new_cases'].to_list()
        try:
            addplot(data, location)
        except Exception as e:
            excluded.append(location)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3, fontsize=6)
    plt.grid('both')
    plt.xlabel('log(ni)')
    plt.ylabel('log(Yi)')
    plt.title("SOC for COVID-19 daily new cases data Series")
    plt.tight_layout()
    plt.draw()
    doc.add_fig()
    doc.add_paragraph("A few countries were excluded from this analysis, because they crashed the soc.py script, " +
                      "these are:" + ", ".join(excluded))

    doc.add_paragraph("Now, we repeat the process for a few select countries")
    plt.figure()
    for location in country_list:
        df_owd = df_all[df_all['location'] == location]
        data = df_owd['new_cases'].to_list()
        try:
            addplot(data, location, ymin=-20)
        except Exception as e:
            pass
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=6)
    plt.grid('both')
    plt.xlabel('log(ni)')
    plt.ylabel('log(Yi)')
    plt.title("SOC for COVID-19 daily new cases data Series in select countries")
    plt.tight_layout()
    plt.draw()
    doc.add_fig()

if __name__ == '__main__':

    p_value = 0.2
    _, data_y = pmodel.pmodel(8192, p_value, slope=0.4)

    soc.soc_plot(data_y, "soc_main chart for p-model series with p = " + str(p_value))

    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report)
    test_report.finish()
    print("Finished ", __file__)
    plt.show()





# Standard imports:
import os
import pandas as pd
import matplotlib.pyplot as plt

# Local imports:
from tools import createdocument, getdata, print_table, specplus, stat


def run(doc, k_means_list):
    doc.add_heading('Exercise 6.2', level=2)
    doc.add_paragraph("Using provided data sets and new cases of COVID-19 in the USA.")
    # open file:
    mount_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount')
    with open(os.path.join(mount_path, 'surftemp504.txt'), 'r') as text_file:
        list1 = [float(x) for x in text_file.read().split('\n')]

    with open(os.path.join(mount_path, 'sol3ghz.dat'), 'r') as text_file:
        list2 = [float(x) for x in text_file.read().split('\n')]

    # Data for USA:
    df_owd = getdata.acquire_data(date_ini='2020-02-20')
    list3 = df_owd['new_cases'].to_list()

    df_table = pd.DataFrame()
    df_table['Data Set'] = ['surftemp504.txt', 'sol3ghz.dat', 'USA new COVID-19 cases']
    # Compute parameters:
    ds = [stat.series2datasetline(data) for data in (list1, list2, list3)]
    df_table['skewness²'] = [x['skewness']**2 for x in ds]
    df_table['kurtosis'] = [x['kurtosis'] for x in ds]
    alpha, beta_t, beta = [specplus.main(data) for data in (list1, list2, list3)]
    plt.close('all')
    df_table[r'$\alpha$'] = alpha
    df_table[r'$\beta$'] = beta
    df_table[r'$\beta_t$'] = beta_t

    if k_means_list is None:
        pass
    else:
        parameter_spaces = (['skewness²', 'kurtosis', r'$\beta$'], ['skewness²', 'kurtosis', r'$\alpha$'])
        for grouping, parameters in zip(k_means_list, parameter_spaces):
            df_table[" x ".join(parameters)] = grouping.predict(df_table[parameters])

    print_table.render_mpl_table(df_table, col_width=3.0, bbox=None, font_size=12)
    doc.add_fig()
    doc.add_paragraph("All datasets would get classified on the same group, if P-Model was included on the " +
                      "training data, since the grouping would basically be dominated by the high kurtosis " +
                      "and skewness of p-model series.\nIt can be seen that the behavior of the US series was " +
                      "clustered along with the colored and non-gaussian noises, which makes sense assuming that " +
                      "the number of new daily cases should be directly proportional to the number active cases, " +
                      "more specifically the number of people in the contagious stage of the disease, which in " +
                      "turn is dependent on the number of new cases of a few preceding days. While this makes sense " +
                      "for more closed regions, the US seems to have a dynamics of the disease spreading at " +
                      "exponential rate initially once it reaches a new state or city, thus the closer similarity to " +
                      "the Non-Gaussian noise behavior instead of the colored noise.\n")
    doc.add_paragraph("Meanwhile, the data series in surftemp504.txt and sol3ghz.dat, are clustered with the random " +
                      "noise in the space with beta, but particularly the sol3ghz.dat gets isolated in its own group " +
                      "when clustering with alpha. Given the correlation between alpha and beta, this might just be " +
                      "an effect of the lack of normalization or scaling prior to applying the K-Means technique. It " +
                      "might also be an effect of the data being")


# Sample execution
if __name__ == '__main__':
    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report, None)
    test_report.finish()
    plt.show()

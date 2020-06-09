
import pandas as pd
import matplotlib.pyplot as plt

from tools import createdocument, getdata, graphs, specplus, stat


def run(doc):
    doc.add_heading("Exercise 6.3", level=2)
    doc.add_paragraph("")
    # Get list of countries:
    df_all = getdata.acquire_data('all')
    country_list = list(set(df_all['location'].to_list()))
    column_list = ['skewness²', 'kurtosis', r'$\alpha$', r'$\beta$', r'$\beta_t$', 'name']
    df = pd.DataFrame(columns=column_list)
    for location in country_list:
        df_owd = df_all[df_all['location'] == location]
        data = df_owd['new_cases'].to_list()
        if len(list(set(data))) > 11:
            ds = stat.series2datasetline(data)
            alpha, beta_t, beta = specplus.main(data)
            plt.close('all')
            if alpha > 0:
                df_temp = pd.DataFrame([[ds['skewness'] ** 2, ds['kurtosis'], alpha, beta, beta_t, location]],
                                       columns=column_list)
                df = df.append(df_temp, ignore_index=True, sort=False)

    # Apply K-Means grouping:
    parameter_spaces = (['skewness²', 'kurtosis', r'$\beta$'], ['skewness²', 'kurtosis', r'$\alpha$'])

    for parameters in parameter_spaces:
        kmeans_obj = graphs.plot_k_means(df, parameters, doc, "COVID-19 New Cases per Country")
        df["beta group" if ("beta" in x for x in parameters) else "alpha group"] = kmeans_obj.labels_

    doc.add_paragraph('Below is the data set created for most countries available in the data collected from' +
                      ' "Out world in Data" website. Only countries with a minimum data diversity were selected, by' +
                      'checking if the number of new cases had assumed at least 12 distinct values. This eliminates ' +
                      'time series that are simply too short or that do not contain enough data richness to apply the' +
                      ' available tools. Also, countries with negative alpha were removed. The maximum silhouette ' +
                      'coefficient method was employed to select the number of clusters, which ended up being 2. The ' +
                      'main group has hte majority of countries, whereas high kurtose, high skeness and high alpha ' +
                      'or beta, which apparently correlates with countries in early stages of the epidemic, where ' +
                      'daily new cases are growing steadily rather than slowing down or retreating.')

    # Write to document.
    doc.add_paragraph(df.to_string())


# Sample execution
if __name__ == '__main__':
    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report)
    test_report.finish()
    plt.show()

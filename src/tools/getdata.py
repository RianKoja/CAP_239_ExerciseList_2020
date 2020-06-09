########################################################################################################################
# Acquire data from Our World in Data repository (local or remote)
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports:
import pandas as pd
import os


# Use this function to yield the dataframe to be analyzed.
def acquire_data(country='United States', date_ini='2020-03-10', date_end='2020-05-28'):
    xls_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount', 'owid-covid-data.xlsx')

    if os.path.isfile(xls_filename):  # Read the csv file:
        df = pd.read_excel(xls_filename)
    else:
        url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
        df = pd.read_csv(url, index_col=0)
        # Save to excel to save time:
        df.to_excel(xls_filename)

    # Separate country data:
    if country != 'all':
        df = df[df.location == country]
    # Separate date interval:
    df = df[df.date.between(date_ini, date_end, inclusive=True)].reset_index()
    # Separate useful columns:
    if country != 'all':
        df = df[['date', 'new_cases']]
    else:
        df = df[['date', 'location', 'new_cases']]
    # df.to_excel('test_df.xlsx') # Use to visualize selected data in excel.
    return df

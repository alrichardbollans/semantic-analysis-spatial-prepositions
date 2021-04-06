import filecmp
import pandas as pd

from Analysis.add_additional_features import additional_features

output_folder = "tests/test outputs/"
archive_folder = "tests/archive folder/"

new_features_not_in_archive = additional_features + ['size_ratio']


def compare_file_hashes(csv1, csv2):
    filecmp.cmp(csv1, csv2)


def get_original_csv(new_csv_file):
    archive_version = new_csv_file.replace(output_folder, archive_folder)
    return archive_version


def generate_dataframes_to_compare(new_csv_file, columns_to_use=None):
    original_csv = get_original_csv(new_csv_file)

    try:
        if columns_to_use is None:
            original_dataframe = pd.read_csv(original_csv)
            new_dataframe = pd.read_csv(new_csv_file)
        else:
            original_dataframe = pd.read_csv(original_csv, usecols=columns_to_use)
            new_dataframe = pd.read_csv(new_csv_file, usecols=columns_to_use)
    except:
        # Some of the csv files are not nicely formed from pandas
        # They need reading differently
        original_dataframe = pd.read_csv(original_csv, header=None, sep='\n')
        new_dataframe = pd.read_csv(new_csv_file, header=None, sep='\n')
        print(original_dataframe)
        print(new_dataframe)

    return new_dataframe, original_dataframe


def dataframe_diff(df1, df2):
    compare_df = df1.eq(df2)
    print(compare_df)

    return compare_df


def dropcolumns_reindexlike(newdf, originaldf):
    """Removes columns in newdf that aren't in orgininaldf and set index of new df to be same as originaldf
    :returns dataframe
    """

    columns_to_check = originaldf.columns.tolist()

    newdf = newdf[columns_to_check]

    newdf = newdf.set_index(originaldf.index)

    return newdf

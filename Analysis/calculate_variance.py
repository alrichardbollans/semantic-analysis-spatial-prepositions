import statistics

import pandas as pd

from Analysis.parition_model import PARTITION_FOLDER
from Analysis.performance_test_functions import OSF_SCORES_FOLDER, BASIC_MODEL_SCORES_FOLDER, NEURAL_MODEL_SCORES_FOLDER

folds_csv = ''


# TODO:output all variances for tests

def get_models_scores(csv_file):
    output_dict = dict()
    folds_df = pd.read_csv(csv_file,index_col=0)
    for model in folds_df.columns.tolist():
        model_scores = folds_df[model].values
        output_dict[model] = model_scores

    return output_dict


def calculate_variance(fold_scores):
    return statistics.variance(fold_scores)


def test():
    scores = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    v = calculate_variance(scores)
    print(v)
    if v != 1.3720238095238095:
        raise ValueError('Something wrong')


def output_variance(table_folder):

    avg_scores = get_models_scores(table_folder + 'avg folds 10fold:10runs.csv')
    ov_scores = get_models_scores(table_folder + 'overall folds 10fold:10runs.csv')

    avg_row = []
    ov_row = []
    for model in avg_scores:
        avgs = avg_scores[model]
        overalls = ov_scores[model]

        sd_of_avgs = statistics.stdev(avgs)
        avg_row.append(sd_of_avgs)
        sd_of_ovs = statistics.stdev(overalls)
        ov_row.append(sd_of_ovs)
        print(model)
        print(sum(avgs) / 100)
        print(sum(overalls) / 100)
        print(sd_of_avgs)
        print(sd_of_ovs)

    headings = list(avg_scores.keys())
    out_df = pd.DataFrame(columns=headings)
    print(out_df)
    rows = [avg_row,ov_row]
    i=0
    for row in rows:
        if i==0:
            row_name = 'Average'
        else:
            row_name = 'Overall'
        i+=1
        dict_to_add = dict()
        for i in range(0, len(headings)):
            dict_to_add[headings[i]] = row[i]

        new_row = pd.Series(data=dict_to_add, name=row_name)
        out_df = out_df.append(new_row,
                       ignore_index=False)

    out_df.to_csv(table_folder + 'fold_standard_deviations.csv')
    print(out_df)


def output_all_variance():
    output_variance(PARTITION_FOLDER + "tables/")



def main():
    output_all_variance()


if __name__ == '__main__':
    main()

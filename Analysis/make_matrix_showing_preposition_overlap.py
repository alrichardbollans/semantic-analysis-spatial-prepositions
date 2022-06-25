import os.path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def prepare_data():
    preposition_list = ['in', 'inside', 'on', 'on top of', 'above', 'over', 'below', 'under', 'against']

    def get_preposition_data(p):
        file = os.path.join('2019 study', 'preposition data', 'semantic-ratio-list' + p + ' .csv')
        data = pd.read_csv(file)
        data['Unique_config'] = data['Scene'] + data['Figure'] + data['Ground']
        return data

    def get_value_for_prep_pair(p1, p2):
        p1_data = get_preposition_data(p1)
        selected_p1 = p1_data[p1_data['selected_atleast_once'] == 1]

        p2_data = get_preposition_data(p2)
        selected_p2 = p2_data[p2_data['selected_atleast_once'] == 1]

        in_either_data =pd.concat([selected_p1,selected_p2])
        in_either_data.drop_duplicates(subset=['Unique_config'], keep='first',inplace=True)
        in_both_data = selected_p1[selected_p1['Unique_config'].isin(selected_p2['Unique_config'])]

        value = float(len(in_both_data.index)) / len(in_either_data.index)

        if p1 == p2:
            assert value==1
        return value


    ## Make df
    prep1_list = []
    prep2_list = []
    values = []
    for p1 in preposition_list:
        for p2 in preposition_list:
            prep1_list.append(p1)
            prep2_list.append(p2)
            values.append(get_value_for_prep_pair(p1,p2))




    out_data = pd.DataFrame({'Prep1':prep1_list,'Prep2':prep2_list,'Values':values})
    file = os.path.join('2019 study', 'preposition data', 'preposition_relations.csv')

    out_data.to_csv(file)


def your_function():
    # TODO: make this look nicer, drop final prepositions in labels
    data_file =os.path.join('2019 study', 'preposition data', 'preposition_relations.csv')

    data = pd.read_csv(data_file)
    result = data.pivot(index='Prep2', columns='Prep1', values='Values')
    # # Generate a mask for the upper triangle
    mask = np.zeros_like(result, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        result,
        mask=mask,
        cmap ="coolwarm",
        # vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    file = os.path.join('2019 study', 'preposition data', 'preposition_relations.png')
    f.tight_layout()
    plt.savefig(file)


def main():
    prepare_data()
    your_function()


if __name__ == '__main__':
    main()

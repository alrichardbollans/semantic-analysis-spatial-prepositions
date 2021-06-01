"""Summary
In general this script compares each pair of configurations checking which better fits the category
 and which is more typical and test the significance of the differences.

 Run process_data.py before this.

"""

import csv
from itertools import combinations
import numpy as np

from baseline_model_testing import GeneratePrepositionModelParameters, PrototypeModel, PREPOSITION_LIST, \
    get_standard_preposition_parameters
from compile_instances import SemanticCollection
from data_import import Configuration, StudyInfo
from polysemy_analysis import DistinctPrototypePolysemyModel, GeneratePolysemeModels

from process_data import ModSemanticData, TypicalityData, UserData

sv_filetag = SemanticCollection.filetag  # Tag for sv task files


def output_2020_study_results(base_folder = None):
    if base_folder is None:
        base_folder = ""
    # 2019 study data allows us to generate a model of typicality
    model_study_info = StudyInfo("2019 study")
    study_info = StudyInfo("2020 study")
    # Begin by loading users
    userdata = UserData(study_info)

    ## data
    typ_data = TypicalityData(userdata)
    svmod_data = ModSemanticData(userdata)

    # Typicality model
    scene_list = model_study_info.scene_name_list
    # First generate baseline model
    features_to_remove = Configuration.object_specific_features.copy()
    preposition_models_dict = get_standard_preposition_parameters()

    baseline_model = PrototypeModel(preposition_models_dict, scene_list, model_study_info,
                                    test_prepositions=PREPOSITION_LIST)
    typ_model = DistinctPrototypePolysemyModel(GeneratePolysemeModels.distinct_model_name, scene_list, scene_list,
                                               model_study_info, test_prepositions=PREPOSITION_LIST,
                                               baseline_model=baseline_model,
                                               features_to_remove=features_to_remove)

    # Object-specific features
    scene_object_functions = {"sv_modtypa1": "", "sv_modtypa2": "Fixed ground. Mobile Figure",
                              "sv_modtypi1": "",
                              "sv_modtypi2": "Ground is container",
                              "sv_modtypi3": "",
                              "sv_modtypi4": "Ground is container",
                              "sv_modtypo1": "Ground is flat container",
                              "sv_modtypo2": "Ground is smaller than figure",
                              "sv_modtypo3": "Ground is convex container",
                              "sv_modtypo4": "Ground is larger than figure",
                              "sv_modtypov1": "Intermediary object between figure and ground",
                              "sv_modtypov2": "Figure is cover and ground is container",
                              "sv_modtypov3": "",
                              "sv_modtypov4": "Functional interaction",
                              "sv_modtypu1": "Ground illuminates figure",
                              "sv_modtypu2": "Intermediary object between figure and ground",
                              "sv_modtypu3": "",
                              "sv_modtypu4": "Functional interaction",

                              }
    # significance level to use
    sig_level = 0.1

    count_number_significant_pairs = 0  # NUmber of pairs where one of the configs is significantly better than the other

    # Get simple config list of tested configurations
    simple_config_list = typ_data.config_list
    # Create config list
    config_list = []
    for simple_config in simple_config_list:
        c = Configuration(simple_config.scene, simple_config.figure, simple_config.ground, study_info)
        config_list.append(c)

    config_pairs = list(combinations(config_list, 2))
    with open(
            base_folder+ study_info.stats_folder + "/"
            + "results.csv",
            "w",
    ) as results_csvfile, open(
        base_folder+ study_info.stats_folder + "/"
        + "disagreements.csv",
        "w",
    ) as disagcsvfile, open(
        base_folder+ study_info.stats_folder + "/"
        + "monotonicity-preserving-examples.csv",
        "w",
    ) as monocsvfile:
        results_writer = csv.writer(results_csvfile)
        disag_writer = csv.writer(disagcsvfile)  # Collect disagreements
        mono_writer = csv.writer(monocsvfile)  # Monotone examples
        # Write headings for csvs
        heading = [
            "Preposition",
            "Configuration 1",
            "Configuration 2",
            "Typicality Scores",
            "Object-Specific Features",
            "Better Category Member",
            "More Typical Configuration"
        ]
        results_writer.writerow(heading)
        disag_writer.writerow(heading)
        mono_writer.writerow(
            [
                "Preposition",
                "Configuration 1",
                "Configuration 2",
                "Configurations Labelled",
                "More Typical Configuration"
            ]
        )

        for preposition in StudyInfo.preposition_list:

            print("Outputting results for:" + str(preposition))

            for pair in config_pairs:
                c1 = pair[0]
                c2 = pair[1]

                if not (c1.configuration_match(c2)):
                    # Only want to output categorisation info on related scenes
                    if typ_data.is_test_config(c1, preposition) and typ_data.is_test_config(c2, preposition):
                        # Get typicality measures
                        value1_array = np.array(c1.row)
                        c1_typ = typ_model.get_typicality(preposition, value1_array, c1.scene, c1.figure, c1.ground,
                                                          study_info)
                        value2_array = np.array(c2.row)
                        c2_typ = typ_model.get_typicality(preposition, value2_array, c2.scene, c2.figure, c2.ground,
                                                          study_info)

                        typ_string = "[" + str(round(c1_typ, 4)) + "],[" + str(round(c2_typ, 4)) + "]"

                        role_string = "[" + scene_object_functions[c1.scene] + "],[" + scene_object_functions[
                            c2.scene] + "]"

                        sv_stat = svmod_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        c1_times_labelled = sv_stat[0]
                        c1_times_not_labelled = sv_stat[1]
                        number_of_tests1 = c1_times_labelled + c1_times_not_labelled
                        if number_of_tests1 != 0:

                            c1_sr = c1_times_labelled / number_of_tests1
                        else:
                            c1_sr = 0
                        c2_times_labelled = sv_stat[2]
                        c2_times_not_labelled = sv_stat[3]

                        number_of_tests2 = c2_times_labelled + c2_times_not_labelled

                        if number_of_tests2 != 0:
                            c2_sr = c2_times_labelled / number_of_tests2
                        else:
                            c2_sr = 0

                        # Work out if monotone example
                        same_categorisation = False
                        configs_labelled = ""
                        if c1_sr == c2_sr:
                            if c1_sr == 1:
                                configs_labelled = "Always"
                                same_categorisation = True
                            if c1_sr == 0:
                                configs_labelled = "Never"
                                same_categorisation = True

                        # Calculate better config and p value

                        p_value = 1
                        if c1_sr == c2_sr:
                            better_config = "None"

                        elif c1_sr > c2_sr:
                            better_config = c1.string_for_cattyp_table()
                            p_value = sv_stat[4]
                        else:
                            sv_stat = svmod_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                            p_value = sv_stat[4]
                            better_config = c2.string_for_cattyp_table()

                        typ_stat = typ_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        c1_selected_over_c2 = typ_stat[1]
                        c2_selected_over_c1 = typ_stat[2]

                        typ_p_value = 1
                        if c1_selected_over_c2 == c2_selected_over_c1:
                            typ_config = "None"
                        elif c1_selected_over_c2 > c2_selected_over_c1:
                            typ_config = c1.string_for_cattyp_table()
                            typ_p_value = typ_stat[3]
                        else:
                            typ_config = c2.string_for_cattyp_table()
                            typ_stat = typ_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                            typ_p_value = typ_stat[3]

                        disagreement = False

                        if better_config != "None" and typ_config != "None" and better_config != typ_config:
                            disagreement = True

                        if p_value <= sig_level:
                            better_config += "*"

                        significantly_more_typical_config = False
                        if typ_p_value <= sig_level:
                            typ_config += "*"
                            significantly_more_typical_config = True

                        if p_value <= sig_level or typ_p_value <= sig_level:
                            count_number_significant_pairs += 1

                        to_write = (
                            [preposition, c1.string_for_cattyp_table(), c2.string_for_cattyp_table(), typ_string,
                             role_string, better_config, typ_config]
                        )

                        results_writer.writerow(to_write)
                        if disagreement:
                            disag_writer.writerow(to_write)

                        if same_categorisation and significantly_more_typical_config:
                            mono_write = (
                                [preposition, c1.string_for_cattyp_table(), c2.string_for_cattyp_table(),
                                 configs_labelled,
                                 typ_config]
                            )
                            mono_writer.writerow(mono_write)

        results_writer.writerow(["Number of significant pairs", str(count_number_significant_pairs)])


if __name__ == '__main__':
    output_2020_study_results()

# This test checks files are correct before testing any models

import os
import unittest
import sys
import tensorflow as tf



sys.path.append('../')

from Analysis.data_import import clean_name
from Analysis.performance_test_functions import Model
from Analysis.baseline_model_testing import PREPOSITION_LIST, GeneratePrepositionModelParameters
from Analysis.classes import Constraint
from Analysis.compile_instances import *

from test_functions import *
import pandas as pd
from pandas._testing import assert_frame_equal

os.chdir('..')


# These tests don't change any files, just check that the values are correct

class MyTestCase(unittest.TestCase):
    # @unittest.skip
    def test_sv_outputs(self):

        study_info = StudyInfo("2019 study")

        svcollection = SemanticCollection(study_info)

        for preposition in svcollection.get_used_prepositions():
            print(preposition)

            new_df, original_df = generate_dataframes_to_compare(
                svcollection.study_info.config_ratio_csv(svcollection.filetag, preposition))

            assert_frame_equal(new_df, original_df)

    def test_comp_outputs(self):
        study_info = StudyInfo("2019 study")
        compcollection = ComparativeCollection(study_info)

        ## Compare constraints in csv with generated constraints
        original_constraint_df = pd.read_csv(archive_folder + study_info.constraint_csv)
        new_constraint_df = pd.read_csv(study_info.constraint_csv)
        to_drop = []
        for f in new_features_not_in_archive:
            to_drop.append(f+"_1")
            to_drop.append(f+"_2")

        new_df_dropped = new_constraint_df.drop(to_drop, axis=1)

        assert_frame_equal(original_constraint_df, new_df_dropped)
        object_property_csv = study_info.input_feature_data_folder + "/object properties.csv"
        property_df = pd.read_csv(object_property_csv)
        for index, row in new_constraint_df.iterrows():
            fig_cont, fig_light = self.get_object_properties(property_df, row.f1)
            fig_cont_standard = study_info.feature_processor.convert_normal_value_to_standardised("figure_container",
                                                                                                  fig_cont)
            fig_light_standard = study_info.feature_processor.convert_normal_value_to_standardised("figure_lightsource",
                                                                                                   fig_light)

            self.assertAlmostEqual(fig_cont_standard, row.figure_container_1)
            self.assertAlmostEqual(fig_light_standard, row.figure_lightsource_1)


            fig_cont, fig_light = self.get_object_properties(property_df, row.f2)
            fig_cont_standard = study_info.feature_processor.convert_normal_value_to_standardised("figure_container",
                                                                                                  fig_cont)
            fig_light_standard = study_info.feature_processor.convert_normal_value_to_standardised("figure_lightsource",
                                                                                                   fig_light)

            self.assertAlmostEqual(fig_cont_standard, row.figure_container_2)
            self.assertAlmostEqual(fig_light_standard, row.figure_lightsource_2)

    # @unittest.skip
    def test_clean_outputs_process_data(self):
        study_info = StudyInfo("2019 study")

        new_clean_sem_csv = study_info.data_folder + "/" + StudyInfo.sem_annotations_name
        new_clean_comp_csv = study_info.data_folder + "/" + StudyInfo.comp_annotations_name
        new_clean_user_csv = study_info.clean_user_csv

        original_clean_sem_csv = archive_folder + new_clean_sem_csv
        original_clean_comp_csv = archive_folder + new_clean_comp_csv
        original_clean_user_csv = archive_folder + new_clean_user_csv

        new_clean_sem_dataset = pd.read_csv(new_clean_sem_csv)
        original_clean_sem_dataset = pd.read_csv(original_clean_sem_csv)

        new_clean_user_dataset = pd.read_csv(new_clean_user_csv)
        original_clean_user_dataset = pd.read_csv(original_clean_user_csv)

        # Doesn't like reading the full csv as different column lengths, so need to specify
        # which columns to use.
        new_clean_comp_dataset = pd.read_csv(new_clean_comp_csv, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
        original_clean_comp_dataset = pd.read_csv(original_clean_comp_csv, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

        assert_frame_equal(new_clean_sem_dataset, original_clean_sem_dataset)
        assert_frame_equal(new_clean_comp_dataset, original_clean_comp_dataset)
        assert_frame_equal(new_clean_user_dataset, original_clean_user_dataset)

    def test_feature_values(self):
        study_info = StudyInfo("2019 study")

        input_feature_csv = study_info.input_feature_csv
        output_path = study_info.feature_output_csv
        means_output_path = study_info.feature_output_folder + "/feature_means.csv"
        std_output_path = study_info.feature_output_folder + "/feature_stds.csv"
        human_readable_path = study_info.human_readable_feature_output_csv

        new_df = pd.read_csv(input_feature_csv)
        new_df_dropped = new_df.drop(additional_features, axis=1)

        original_df = pd.read_csv(archive_folder + input_feature_csv)

        assert_frame_equal(new_df_dropped, original_df)

        new_df = pd.read_csv(output_path)
        print(new_df)
        new_df_dropped = new_df.drop(new_features_not_in_archive, axis=1)
        print(new_df_dropped)
        original_df = pd.read_csv(archive_folder + output_path)
        assert_frame_equal(new_df_dropped, original_df)
        # new_df, original_df = generate_dataframes_to_compare(means_output_path)
        #
        # assert_frame_equal(new_df, original_df)
        # new_df, original_df = generate_dataframes_to_compare(std_output_path)
        #
        # assert_frame_equal(new_df, original_df)
        new_df = pd.read_csv(human_readable_path)
        new_df_dropped = new_df.drop(new_features_not_in_archive, axis=1)
        original_df = pd.read_csv(archive_folder + human_readable_path)
        assert_frame_equal(new_df_dropped, original_df)

    @staticmethod
    def get_object_properties(property_df, obj):
        class_name = clean_name(obj)
        # print(property_df)
        index = property_df.index[property_df.Object == class_name]

        cont = property_df.iloc[index]["CN_ISA_CONTAINER"].values[0]
        light = property_df.iloc[index]["CN_UsedFor_Light"].values[0]

        # print(f"name: {class_name}. cont: {cont}. light: {light}")

        return cont, light

    def test_os_agreement(self):
        study_info = StudyInfo("2019 study")

        input_feature_csv = study_info.input_feature_csv
        object_property_csv = study_info.input_feature_data_folder + "/object properties.csv"
        human_readable_path = study_info.human_readable_feature_output_csv

        new_df = pd.read_csv(input_feature_csv)
        property_df = pd.read_csv(object_property_csv)
        for index, row in new_df.iterrows():
            fig_cont, fig_light = self.get_object_properties(property_df, row.Figure)
            grd_cont, grd_light = self.get_object_properties(property_df, row.Ground)

            self.assertEqual(fig_cont, row.figure_container)
            self.assertEqual(fig_light, row.figure_lightsource)

            self.assertEqual(grd_cont, row.ground_container)
            self.assertEqual(grd_light, row.ground_lightsource)

            # print(f"{fig_cont}:{row.figure_container}")
            # print(f"{fig_light}:{row.figure_lightsource}")
            # print(f"{grd_cont}:{row.ground_container}")
            # print(f"{grd_light}:{row.ground_lightsource}")
        new_df = pd.read_csv(human_readable_path)
        for index, row in new_df.iterrows():
            fig_cont, fig_light = self.get_object_properties(property_df, row.Figure)
            grd_cont, grd_light = self.get_object_properties(property_df, row.Ground)

            self.assertEqual(fig_cont, row.figure_container)
            self.assertEqual(fig_light, row.figure_lightsource)

            self.assertEqual(grd_cont, row.ground_container)
            self.assertEqual(grd_light, row.ground_lightsource)

    def test_specs(self):

        self.assertEqual(pd.__version__, '1.0.3')
        self.assertEqual(tf.__version__,'2.4.0')

    def test_remove_features(self):
        study_info = StudyInfo("2019 study")
        scene_list = study_info.scene_name_list
        features_to_remove = Configuration.object_specific_features.copy()
        features = []
        [features.append(f) for f in study_info.all_feature_keys if f not in features_to_remove]
        generic_model = Model('generic', scene_list,study_info)

        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in PREPOSITION_LIST:
            M = GeneratePrepositionModelParameters(study_info, p, scene_list,
                                                   features_to_remove=features_to_remove)
            preposition_models_dict[p] = M

        for p in PREPOSITION_LIST:


            for idx, row in preposition_models_dict[p].allFeatures.iterrows():
                row_with_features = row
                for idx2, row2 in preposition_models_dict[p].feature_dataframe.iterrows():
                    if idx == idx2:

                        row_without_features = row2
                        break

                removed_features_array = generic_model.remove_features_from_array(row_with_features,features_to_remove)

                self.assertEqual(list(row_without_features),removed_features_array)

if __name__ == '__main__':
    unittest.main()

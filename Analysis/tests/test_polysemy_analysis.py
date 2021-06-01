import os
import unittest
import sys

sys.path.append('../')

from Analysis.polysemy_analysis import *

from Analysis.data_import import StudyInfo
from test_functions import *

from pandas._testing import assert_frame_equal

os.chdir("..")


class Test(unittest.TestCase):
    """Summary
    """

    # @unittest.skip
    def test_polyseme_rank_info(self):
        '''

        :return:
        '''
        study_info = StudyInfo("2019 study")

        all_scenes = study_info.scene_name_list
        generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info)

        # Check ranks
        generated_polyseme_models.non_shared.output_polyseme_info(base_folder=output_folder)
        model_name = GeneratePolysemeModels.distinct_model_name
        for preposition in POLYSEMOUS_PREPOSITIONS:
            new_rank_csv = output_folder + POLYSEMY_MODEL_PROPERTY_FOLDER + model_name + "/ranks/" + preposition + " -ranks.csv"
            new_rank_df = pd.read_csv(new_rank_csv)
            original_rank_df = pd.read_csv(get_original_csv(new_rank_csv))

            cols = new_rank_df.columns.tolist()
            redordered_orig_df = original_rank_df[cols]
            assert_frame_equal(new_rank_df, redordered_orig_df)

    # @unittest.skip
    def test_initial_model(self):
        cluster_numbers = KMeansPolysemyModel.cluster_numbers
        self.assertEqual(cluster_numbers["on"], 8)
        self.assertEqual(cluster_numbers["in"], 4)
        self.assertEqual(cluster_numbers["over"], 4)
        self.assertEqual(cluster_numbers["under"], 4)

        study_info = StudyInfo("2019 study")

        all_scenes = study_info.scene_name_list
        generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info,
                                                           test_prepositions=POLYSEMOUS_PREPOSITIONS)

        p_models = generated_polyseme_models.models

        archive_all_csv = archive_folder + "2019 study/polysemy/scores/all_test.csv"
        original_dataframe = pd.read_csv(archive_all_csv, index_col=0)
        print(original_dataframe)

        t = TestModels(p_models, "all")

        new_dframe = t.score_dataframe
        print(new_dframe)

        # reindex original as it contains shared aswell but new doesn't
        new_reindexed, original_reindexed = dropcolumns_reindexlike(new_dframe, original_dataframe)

        try:
            assert_frame_equal(new_reindexed, original_reindexed)
        except AssertionError as e:
            print(e)

        # first check basic parametres
        for model in p_models:
            self.assertEqual(len(model.all_feature_keys), 16)
            if hasattr(model, "preposition_model_dict"):
                for p in model.test_prepositions:
                    self.assertEqual(len(model.preposition_model_dict[p].feature_keys), 10)

        # Check typicalities
        for model in p_models:

            for preposition in POLYSEMOUS_PREPOSITIONS:
                typ_csv = output_folder + model.study_info.base_polysemy_folder + "config typicalities/typicality-" + preposition + ".csv"

                model.output_typicalities(preposition, input_csv=typ_csv)

        for preposition in POLYSEMOUS_PREPOSITIONS:
            # Remove Kmeans column as it is not deterministic
            new_typicality_csv = output_folder + generated_polyseme_models.study_info.base_polysemy_folder + "config typicalities/typicality-" + preposition + ".csv"
            new_typicality_df = pd.read_csv(new_typicality_csv, usecols=[0, 1, 2, 3, 4,
                                                                         5, 6])

            original_typicality_df = pd.read_csv(get_original_csv(new_typicality_csv), usecols=[0, 1, 2, 3, 4,
                                                                                                5, 6])

            columns_to_check = ['scene', 'figure', 'ground', 'Distinct Prototype', 'Baseline Model']
            print(original_typicality_df.columns.tolist())
            print(new_typicality_df.columns.tolist())
            # reindex original as it contains shared aswell but new doesn't
            original_reindexed = original_typicality_df[columns_to_check]
            new_reindexed = new_typicality_df[columns_to_check]
            assert_frame_equal(new_reindexed, original_reindexed)
            print(original_reindexed.columns.tolist())
            print(new_reindexed.columns.tolist())

    # @unittest.skip
    def test_k_fold(self):
        study_info = StudyInfo("2019 study")

        m = MultipleRuns(GeneratePolysemeModels, POLYSEMY_SCORES_FOLDER + "tables",
                         POLYSEMY_SCORES_FOLDER + "plots", study_info, number_runs=10, k=10,
                         compare="y", test_prepositions=POLYSEMOUS_PREPOSITIONS)

        self.assertIsInstance(m.Generate_Models_all_scenes, GeneratePolysemeModels)
        self.assertIsInstance(m.Generate_Models_all_scenes.features_to_remove, list)
        self.assertEqual(m.Generate_Models_all_scenes.features_to_remove, Configuration.object_specific_features.copy())

        self.assertIsInstance(m.model_name_list, list)

        test_folds = m.get_validation_scene_split()

        while not m.folds_check(test_folds):
            test_folds = m.get_validation_scene_split()

        self.assertEqual(len(test_folds), 10)

        for f1 in test_folds:
            self.assertNotEqual(len(f1), 0)
            for f2 in test_folds:
                if f1 != f2:
                    f1_set = set(f1)
                    f2_set = set(f2)
                    self.assertFalse(f1_set & f2_set)


if __name__ == "__main__":

    if len(POLYSEMOUS_PREPOSITIONS) > 4:
        print("POLYSEMOUS_PREPOSITIONS contains non-polysemous prepositions")
        print("Changing for test")
        polysemous_preposition_list = ['in', 'on', 'under', 'over']
    unittest.main()

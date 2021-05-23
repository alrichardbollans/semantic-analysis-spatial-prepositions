import sys



sys.path.append('../')

import unittest
import os
from Analysis.performance_test_functions import TestModels
from Analysis.baseline_model_testing import *
from test_functions import *

from pandas._testing import assert_frame_equal
import numpy as np

os.chdir("..")


class Test(unittest.TestCase):
    """Summary
    """
    def t_multiple_runs(self, m_runs):
        i = 0
        while i < 10:
            i += 1
            folds = m_runs.get_validation_scene_split()

            if m_runs.folds_check(folds):
                for f in folds:
                    print(("Fold Number:" + str(folds.index(f))))
                    test_scenes = f
                    train_scenes = []
                    for train_fold in folds:
                        if train_fold != f:
                            for t in train_fold:
                                train_scenes.append(t)
                    train_set = set(train_scenes)
                    self.assertFalse(any(x in train_set for x in test_scenes))

    # @unittest.skip
    def test_initial_model(self):
        study_info = StudyInfo("2019 study")

        original_dataframe = pd.read_csv(archive_folder+'2019 study/scores/tables/all-model scores.csv', index_col=0)

        m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                         BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info, number_runs=1, k=10)
        self.t_multiple_runs(m)

        generate_models = m.Generate_Models_all_scenes
        models = generate_models.models

        # Check models are using 10 features
        first_model = models[0]
        self.assertEqual(len(first_model.all_feature_keys), 16)
        for p in first_model.test_prepositions:
            self.assertEqual(len(first_model.preposition_model_dict[p].feature_keys), 10)

        t = TestModels(models, "all")
        new_dframe = t.score_dataframe

        new_reindexed, original_reindexed = dropcolumns_reindexlike(new_dframe, original_dataframe)

        print(original_reindexed)
        print(new_reindexed)

        assert_frame_equal(new_reindexed, original_reindexed)

    # @unittest.skip
    def test_k_fold(self):
        study_info = StudyInfo("2019 study")

        m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                         BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info, number_runs=20, k=2, compare="y")

        self.assertIsInstance(m.Generate_Models_all_scenes, GenerateBasicModels)
        self.assertIsInstance(m.Generate_Models_all_scenes.features_to_remove, list)
        self.assertEqual(m.Generate_Models_all_scenes.features_to_remove, Configuration.object_specific_features.copy())

        self.assertIsInstance(m.model_name_list, list)

        test_folds = m.get_validation_scene_split()

        while not m.folds_check(test_folds):
            test_folds = m.get_validation_scene_split()

        self.assertEqual(len(test_folds), 2)
        for f1 in test_folds:
            self.assertNotEqual(len(f1), 0)
            for f2 in test_folds:
                if f1 != f2:
                    f1_set = set(f1)
                    f2_set = set(f2)
                    self.assertFalse(f1_set & f2_set)

    # @unittest.skip
    def test_removing_features(self):

        study_info = StudyInfo("2019 study")
        functional_features = ["location_control", "support"]
        m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                         BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info, number_runs=20, k=2, features_to_test=functional_features)
        self.assertIsInstance(m.Generate_Models_all_scenes, GenerateBasicModels)
        self.assertIsInstance(m.Generate_Models_all_scenes.features_to_remove, list)
        self.assertEqual(m.Generate_Models_all_scenes.features_to_remove, Configuration.object_specific_features.copy())

        self.assertIsInstance(m.model_name_list, list)
        self.assertIsInstance(m.features_to_test, list)

        test_folds = m.get_validation_scene_split()

        self.assertEqual(len(test_folds), 2)
        for f1 in test_folds:
            self.assertNotEqual(len(f1), 0)
            for f2 in test_folds:
                if f1 != f2:
                    f1_set = set(f1)
                    f2_set = set(f2)
                    self.assertFalse(f1_set & f2_set)

        without_lc_model = \
        m.generate_models(m.scene_list, m.scene_list, extra_features_to_remove=["location_control"]).models[0]
        without_sup_model = m.generate_models(m.scene_list, m.scene_list, extra_features_to_remove=["support"]).models[
            0]

        all_feature_keys = without_lc_model.preposition_model_dict['in'].all_feature_keys

        for p in PREPOSITION_LIST:
            for feature in all_feature_keys:
                if feature == "location_control" or feature in Configuration.object_specific_features.copy():
                    self.assertEqual(
                        without_lc_model.preposition_model_dict[p].regression_weights[all_feature_keys.index(feature)],
                        0)

                if feature == "support" or feature in Configuration.object_specific_features.copy():
                    self.assertEqual(
                        without_sup_model.preposition_model_dict[p].regression_weights[all_feature_keys.index(feature)],
                        0)

        for p in PREPOSITION_LIST:
            lc_test_array = without_lc_model.preposition_model_dict[p].prototype.copy()
            lc_test_array[all_feature_keys.index("location_control")] += -5

            sup_test_array = without_sup_model.preposition_model_dict[p].prototype.copy()
            sup_test_array[all_feature_keys.index("support")] += -5

            self.assertEqual(without_lc_model.get_typicality(p, lc_test_array), 1)
            self.assertEqual(without_sup_model.get_typicality(p, sup_test_array), 1)

    # @unittest.skip
    def test_semantic_distance(self):
        study_info = StudyInfo("2019 study")
        d = SemanticMethods.semantic_distance(np.array([1, 0.75]), np.array([1, 1]), np.array([2, 3]))

        self.assertIsInstance(d, float)
        self.assertEqual(d, 2)

    def test_semantic_similarity(self):
        study_info = StudyInfo("2019 study")
        d = SemanticMethods.semantic_similarity(np.array([1, 0.75]), np.array([1, 1]), np.array([2, 3]))

        self.assertIsInstance(d, float)
        self.assertAlmostEqual(d, 0.1353352, places=5)

    # @unittest.skip
    def test_model_parameters(self):
        study_info = StudyInfo("2019 study")

        m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                         BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info)

        generate_models = m.Generate_Models_all_scenes
        models = generate_models.models

        our_model = models[0]

        # Check we've got the correct model
        self.assertEqual(our_model.name, PrototypeModel.name)

        for p in our_model.test_prepositions:
            # Check regression weights
            original_regressions_csv = archive_folder + our_model.preposition_model_dict[p].regression_weight_csv
            original_weight_df = pd.read_csv(original_regressions_csv, index_col=0)

            for feature in our_model.preposition_model_dict[p].feature_keys:
                value = original_weight_df.at[feature, '0']
                self.assertAlmostEqual(value, self.get_feature_weight(our_model, p, feature), places=7)

            # Check prototypes
            original_prototype_csv = archive_folder + our_model.preposition_model_dict[p].prototype_csv
            original_prototype_df = pd.read_csv(original_prototype_csv, index_col=0)

            for feature in our_model.preposition_model_dict[p].feature_keys:
                value = original_prototype_df.at[feature, '0']
                self.assertAlmostEqual(value, self.get_feature_prototype(our_model, p, feature), places=7)

    @staticmethod
    def get_feature_prototype(our_model, preposition, feature):
        p_model = our_model.preposition_model_dict[preposition]
        feature_index = p_model.get_feature_index_in_array(feature)
        feature_value = p_model.prototype[feature_index]
        return feature_value

    @staticmethod
    def get_feature_weight(our_model, preposition, feature):
        p_model = our_model.preposition_model_dict[preposition]
        feature_index = p_model.get_feature_index_in_array(feature)
        feature_value = p_model.regression_weights[feature_index]

        return feature_value

    def check_feature_weight_value(self, our_model, preposition, feature, value):
        feature_value = self.get_feature_weight(our_model, preposition, feature)
        print(feature_value)
        self.assertAlmostEqual(value, feature_value, places=2)


if __name__ == "__main__":
    unittest.main()

import sys

sys.path.append('../')

import os
import unittest

from Analysis.neural_models import *

os.chdir("..")

study_info = StudyInfo("2019 study")
scene_list = study_info.scene_name_list
features_to_remove = Configuration.object_specific_features.copy()
preposition_models_dict = get_standard_preposition_parameters(scene_list)


class Test(unittest.TestCase):
    """Summary
    """

    def test_dataframe_conversion(self):
        dnn_model = NeuralNetworkCategorisationModel(preposition_models_dict, scene_list, study_info,
                                                     train_test_proportion=0.8,
                                                     number_of_epochs=1)
        # Test: oversampling, dropping nonfeatures, dropping unused features,
        for p in PREPOSITION_LIST:
            for features_tensor, target_tensor in dnn_model.training_data_dict[p]:
                number_of_features = len(dnn_model.study_info.all_feature_keys) - len(
                    Configuration.object_specific_features)

                print(f'features:{features_tensor} target:{target_tensor}')
                self.assertEqual(number_of_features, len(features_tensor[0]))
                self.assertLessEqual(target_tensor[0], 1)

                # Matching labels
                match = False
                for idx, row in preposition_models_dict[p].feature_dataframe.iterrows():
                    if list(features_tensor[0]) == list(row):
                        match = True
                        break

                self.assertTrue(match)

                # Oversampling
                if target_tensor[0] > 0:
                    copies = [f_tensor for f_tensor, t_tensor in dnn_model.training_data_dict[p] if
                              list(f_tensor[0]) == list(features_tensor[0])]
                    self.assertGreater(len(copies), 2)

    def test_dataset_prep(self):
        sup_model = SupervisedNeuralTypicalityModel(scene_list, scene_list, study_info, features_to_remove,
                                                    train_test_proportion=0.8, number_of_epochs=1)

        for p in PREPOSITION_LIST:
            print(f"Starting: {p}")
            train_constraints = sup_model.get_test_constraints(p)

            for features_tensor, target_tensor in sup_model.train_datasets[p]:
                number_of_features = (len(sup_model.study_info.all_feature_keys) - len(
                    Configuration.object_specific_features))

                # print(f'features:{features_tensor} target:{target_tensor}')
                self.assertEqual(number_of_features, len(features_tensor[0]))
                self.assertLessEqual(target_tensor[0], 1)

            # Oversampling
            for c in train_constraints:
                train_array = np.subtract(c.lhs_values, c.rhs_values)
                train_array = sup_model.remove_features_from_array(train_array, sup_model.features_to_remove)
                train_array2 = [-x for x in train_array]

                copies = [f_tensor for f_tensor, t_tensor in sup_model.train_datasets[p] if
                          list(f_tensor[0]) == train_array or list(f_tensor[0]) == train_array2]
                self.assertGreaterEqual(len(copies), c.weight)

    def test_generator(self):
        study_info = StudyInfo("2019 study")

        m = MultipleRuns(GenerateNeuralModels, "tests",
                         "tests", study_info, test_prepositions=PREPOSITION_LIST,
                         number_runs=1,
                         k=2,
                         compare="y")
        neural_categorisation = m.Generate_Models_all_scenes.neural_categorisation
        # neural_supervised = m.Generate_Models_all_scenes.neural_supervised
        self.assertEqual(neural_categorisation.number_of_epochs, 200)
        # self.assertEqual(neural_supervised.number_of_epochs, 200)
        self.assertEqual(neural_categorisation.train_test_proportion, 1)
        # self.assertEqual(neural_supervised.train_test_proportion, 1)
        # self.assertGreater(len(neural_supervised.callbacks), 0)
        self.assertGreater(len(neural_categorisation.callbacks), 0)

    def test_typicality_output(self):
        study_info = StudyInfo("2019 study")

        m = MultipleRuns(GenerateNeuralModels, "tests",
                         "tests", study_info, test_prepositions=PREPOSITION_LIST,
                         number_runs=1,
                         k=2,
                         compare="y")
        neural_categorisation = m.Generate_Models_all_scenes.neural_categorisation

        config_list = study_info.config_list
        c = config_list[0]

        value_array = np.array(c.row)
        typicality = neural_categorisation.get_typicality('in', value_array, scene=c.scene, figure=c.figure,
                                                          ground=c.ground, study=study_info)
        print(type(typicality))
        self.assertIsInstance(typicality, float)


if __name__ == "__main__":
    unittest.main()

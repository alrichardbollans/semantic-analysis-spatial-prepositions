import copy
import random

from sklearn.model_selection import train_test_split

from Analysis.baseline_model_testing import GenerateBasicModels
from performance_test_functions import PREPOSITION_LIST, compare_models

from Analysis.polysemy_analysis import DistinctPrototypePolysemyModel, SalientFeature, GeneratePolysemeModels

PARTITION_FOLDER = "model evaluation/partition/"


class DataPartitionPolysemyModel(DistinctPrototypePolysemyModel):
    # THis model helps to check if arbitrailiy partitioning the data improves the baseline prototpe model
    name = "Partition Model"

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST,
                 preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, DataPartitionPolysemyModel.name, train_scenes, test_scenes,
                                                study_info_, test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by finding median feature values of good instances.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """
        if preposition in self.test_prepositions:
            # Find value of feature such that half of preposition instances are greater and half are less than value
            number_of_features = len(original_salient_features)

            candidate_features = []

            # Get new salient features
            for feature in self.study_info.all_feature_keys:
                if feature not in self.features_to_remove:
                    if not any(x.name == feature for x in original_salient_features):
                        candidate_features.append(feature)

            new_features = random.choices(candidate_features, k=number_of_features)

            new_salient_features = []
            for f in new_features:
                median = self.baseline_model.preposition_model_dict[preposition].goodAllFeatures[f].median()

                new_f = SalientFeature(f, median, "g")
                new_salient_features.append(new_f)

            new_polysemes = self.generate_polysemes(preposition, new_salient_features)
            return new_polysemes
        else:
            return None


class DistinctPrototypeMedianPolysemyModel(DistinctPrototypePolysemyModel):
    name = "Median Model"

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST,
                 preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, self.name, train_scenes, test_scenes, study_info_,
                                                test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by finding median feature values of good instances.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """

        # Find value of feature such that half of preposition instances are greater and half are less than value
        if preposition in self.test_prepositions:
            new_salient_features = []
            for f in original_salient_features:
                new_f = copy.deepcopy(f)

                median = self.baseline_model.preposition_model_dict[preposition].goodAllFeatures[new_f.name].median()
                new_f.value = median

                new_salient_features.append(new_f)

            new_polysemes = self.generate_polysemes(preposition, new_salient_features)
            return new_polysemes
        else:
            return None


class GeneratePartitionModels(GenerateBasicModels):

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST,
                 preserve_empty_polysemes=False):
        GenerateBasicModels.__init__(self, train_scenes, test_scenes, study_info_,
                                     test_prepositions=test_prepositions)

        self.partition_model = DataPartitionPolysemyModel(train_scenes,
                                                          test_scenes, study_info_,
                                                          baseline_model=self.our_model,
                                                          features_to_remove=self.features_to_remove)

        # self.median_model = DistinctPrototypeMedianPolysemyModel(train_scenes, test_scenes, study_info_,
        #                                                          baseline_model=self.our_model,
        #                                                          features_to_remove=self.features_to_remove)

        self.generate_model_lists()


def test_models():
    compare_models(10, 10, GeneratePartitionModels, PARTITION_FOLDER, test_prepositions=PREPOSITION_LIST)


def main():
    test_models()


if __name__ == '__main__':
    main()

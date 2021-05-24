from Analysis.baseline_model_testing import PREPOSITION_LIST, GeneratePrepositionModelParameters, \
    PrototypeModel
from Analysis.performance_test_functions import ModelGenerator, TestModels, compare_models, OSF_SCORES_FOLDER
from Analysis.neural_models import NeuralNetworkCategorisationModel
from Analysis.data_import import Configuration, StudyInfo
from Analysis.polysemy_analysis import DistinctPrototypePolysemyModel, SalientFeature, GeneratePolysemeModels

OS_FEATURES_TO_REMOVE = ["ground_verticality"]

ASSOCIATED_FEATURES = {"in": ["ground_container"], "on": ["ground_container", "size_ratio"],
                       "under": ["figure_container", "ground_lightsource"],
                       "over": ["ground_container", "figure_lightsource"], "against": ["size_ratio"]}
ASSOCIATED_FEATURES["inside"] = ASSOCIATED_FEATURES["in"]
ASSOCIATED_FEATURES["on top of"] = ASSOCIATED_FEATURES["on"]
ASSOCIATED_FEATURES["below"] = ASSOCIATED_FEATURES["under"]
ASSOCIATED_FEATURES["above"] = ASSOCIATED_FEATURES["over"]


class OSFSenseModel(DistinctPrototypePolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST,
                 preserve_empty_polysemes=False,
                 baseline_model=None):
        features_to_remove = Configuration.object_specific_features.copy()
        DistinctPrototypePolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove,
                                                oversample=True)

    def get_non_shared_prototype_polyseme_dict(self):
        """Summary

        Returns:
            TYPE: Description
        """
        out = dict()

        fig_container_value = self.feature_processer.convert_normal_value_to_standardised("figure_container", 0.5)
        ground_container_value = self.feature_processer.convert_normal_value_to_standardised("ground_container", 0.5)

        fig_mobile_value = self.feature_processer.convert_normal_value_to_standardised("size_ratio", 1)

        fig_lightsource_value = self.feature_processer.convert_normal_value_to_standardised("figure_lightsource", 0.5)
        ground_lightsource_value = self.feature_processer.convert_normal_value_to_standardised("ground_lightsource",
                                                                                               0.5)

        # On
        f1 = SalientFeature("ground_container", ground_container_value, "l")
        f2 = SalientFeature("size_ratio", fig_mobile_value, "l")
        on_salient_features = [f1, f2]

        out["on"] = self.refine_ideal_meaning("on", on_salient_features)
        out["on top of"] = self.refine_ideal_meaning("on top of", on_salient_features)

        # In
        f1 = SalientFeature("ground_container", ground_container_value, "g")

        in_salient_features = [f1]

        out["in"] = self.refine_ideal_meaning("in", in_salient_features)
        out["inside"] = self.refine_ideal_meaning("inside", in_salient_features)

        # Under
        f1 = SalientFeature("figure_container", fig_container_value, "g")
        f2 = SalientFeature("ground_lightsource", ground_lightsource_value, "g")

        under_salient_features = [f1, f2]

        out["under"] = self.refine_ideal_meaning("under", under_salient_features)
        out["below"] = self.refine_ideal_meaning("below", under_salient_features)

        # Over
        f1 = SalientFeature("ground_container", ground_container_value, "g")
        # There's no instance of this in sv dataset, so remove.
        # f2 = SalientFeature("figure_lightsource", fig_lightsource_value, "g")

        over_salient_features = [f1]  # , f2]

        out["over"] = self.refine_ideal_meaning("over", over_salient_features)
        out["above"] = self.refine_ideal_meaning("above", over_salient_features)

        # against

        f1 = SalientFeature("size_ratio", fig_mobile_value, "l")

        against_salient_features = [f1]

        out["against"] = self.refine_ideal_meaning("against", against_salient_features)

        return out


class GenerateOSModels(GeneratePolysemeModels):

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions):
        GeneratePolysemeModels.__init__(self, train_scenes, test_scenes, study_info_, test_prepositions)
        # self.features_to_remove = OS_FEATURES_TO_REMOVE
        # preposition_models_dict_with_OSF = dict()
        #
        # # Get parameters for each preposition
        # for p in PREPOSITION_LIST:
        #     # Only include associated OSFs
        #     features_to_remove = [x for x in Configuration.object_specific_features if x not in ASSOCIATED_FEATURES[p]]
        #     M = GeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
        #                                            features_to_remove=features_to_remove)
        #     M.work_out_models()
        #     preposition_models_dict_with_OSF[p] = M
        #
        # self.neural_categorisation = NeuralNetworkCategorisationModel(preposition_models_dict_with_OSF,
        #                                                               self.test_scenes,
        #                                                               self.study_info)
        # self.neural_categorisation.name = "Neural Net Classification (OS)"
        #
        # self.baseline_model_with_osf = PrototypeModel(preposition_models_dict_with_OSF, self.test_scenes,
        #                                               self.study_info)
        # self.baseline_model_with_osf.name = "Baseline (w. OSF)"

        self.sense_model = OSFSenseModel("Sense Model (OS)", train_scenes, self.test_scenes, self.study_info,
                                         self.test_prepositions, baseline_model=self.baseline_model)

        self.generate_model_lists()


def output_model_params():
    s_info = StudyInfo("2019 study")
    all_scenes = s_info.scene_name_list

    generated_polyseme_models = GenerateOSModels(all_scenes, all_scenes, s_info, PREPOSITION_LIST)

    sense_model = generated_polyseme_models.sense_model

    sense_model.output_polyseme_info()


def test_models():
    """Summary

    Args:
        study_info_ (TYPE): Description
    """

    compare_models(10, 10, GenerateOSModels, OSF_SCORES_FOLDER, test_prepositions=PREPOSITION_LIST)


def main():
    output_model_params()
    # test_models()


if __name__ == '__main__':
    main()

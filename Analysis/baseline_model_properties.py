import matplotlib as mpl

from Analysis.baseline_model_testing import GenerateBasicModels, PREPOSITION_LIST
from Analysis.data_import import StudyInfo


def plot_feature_regression(study_info):
    mpl.rcParams['axes.titlesize'] = 'xx-large'
    mpl.rcParams['axes.labelsize'] = 'xx-large'

    scene_list = study_info.scene_name_list
    generated_models = GenerateBasicModels(scene_list, scene_list, study_info)

    Minside = generated_models.preposition_parameters_dict["inside"]
    Minside.plot_single_feature_regression("bbox_overlap_proportion")


def plot_feature_spaces(study_info):
    scene_list = study_info.scene_name_list
    generated_models = GenerateBasicModels(scene_list, scene_list, study_info)

    Min = generated_models.preposition_parameters_dict["in"]
    Min.plot_feature_space("bbox_overlap_proportion", "location_control")

    Mon = generated_models.preposition_parameters_dict["on"]
    Mon.plot_feature_space("support", "contact_proportion")
    Mon.plot_feature_space("support", "above_proportion")

    Minside = generated_models.preposition_parameters_dict["inside"]
    Minside.plot_feature_space("bbox_overlap_proportion", "location_control")


def plot_preposition_graphs(study_info):
    """Summary

    Args:
        study_info (TYPE): Description
    """
    scene_list = study_info.scene_name_list
    generated_models = GenerateBasicModels(scene_list, scene_list, study_info)

    for p in PREPOSITION_LIST:

        M = generated_models.preposition_parameters_dict[p]
        # M.output_models()
        M.plot_models()


if __name__ == "__main__":
    study_info = StudyInfo("2019 study")

    plot_preposition_graphs(study_info)
    mpl.rcParams['font.size'] = 25
    plot_feature_regression(study_info)
    plot_feature_spaces(study_info)


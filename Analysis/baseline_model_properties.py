import matplotlib as mpl

from Analysis.baseline_model_testing import GenerateBasicModels, preposition_list


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

    for p in preposition_list:
        M = generated_models.preposition_parameters_dict[p]
        M.output_models()
        M.plot_models()

if __name__ == "main":
    mpl.rcParams['font.size'] = 25
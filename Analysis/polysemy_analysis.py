"""Summary
This file provides classes for generating models of typicality and running tests on them.
First run compile_instances.py
Attributes:
    COMP_FILETAG (str): Description
    NON_POLYSEMOUS_PREPOSITIONS (list): Description
    POLYSEMOUS_PREPOSITIONS (list): Description
    preposition_list (TYPE): Description
    SV_FILETAG (str): Description

"""
#

# Standard imports
import copy

import pandas as pd
import numpy as np
import itertools

# Ml modules

from sklearn.cluster import KMeans

from Analysis.neural_models import NeuralNetworkCategorisationModel
from baseline_model_testing import GeneratePrepositionModelParameters, SemanticMethods, PrototypeModel, PREPOSITION_LIST
from Analysis.performance_test_functions import ModelGenerator, TestModels, MultipleRuns, Model, POLYSEMY_SCORES_FOLDER, \
    ALL_PREPS_POLYSEMY_SCORES_FOLDER, compare_models, POLYSEMY_MODEL_PROPERTY_FOLDER
from data_import import Configuration, StudyInfo
from compile_instances import SemanticCollection, ComparativeCollection

# Useful global variables
SV_FILETAG = SemanticCollection.filetag  # Tag for sv task files
COMP_FILETAG = ComparativeCollection.filetag  # Tag for comp task files

POLYSEMOUS_PREPOSITIONS = ['in', 'on', 'under', 'over']  # list of prepositions which exist in the data
NON_POLYSEMOUS_PREPOSITIONS = ["inside", "above", "below", "on top of", 'against']


class ClusterInModel:
    """Summary
    
    Attributes:
        centre (TYPE): Description
        preposition (TYPE): Description
        rank (TYPE): Description
        weights (TYPE): Description
    """

    def __init__(self, preposition, centre, weights, rank):
        """Summary
        
        Args:
            preposition (TYPE): Description
            centre (TYPE): Description
            weights (TYPE): Description
            rank (TYPE): Description
        """
        self.preposition = preposition
        self.centre = centre
        self.weights = weights
        self.rank = rank


class SalientFeature:
    """Summary

    Attributes:
        feature (TYPE): Description
        gorl (TYPE): Description
        value (TYPE): Description
    """

    def __init__(self, name, value, gorl):
        """Summary

        Args:
            feature (TYPE): Description
            value (TYPE): Description
            gorl (TYPE): Description
        """
        self.name = name
        self.value = value
        self.gorl = gorl


class Polyseme:
    """Summary
    Polyseme is defined by feature values: a configuration can be a polyseme instance if certain
    conditions on the feature values are satisfied.
    This class uses GeneratePrepositionModelParameters to find feature weights and prototypes for the polyseme.
    Attributes:
        annotation_csv (TYPE): Description
        eq_feature_dict (TYPE): Description
        greater_feature_dict (TYPE): Description
        less_feature_dict (TYPE): Description
        mean_csv (TYPE): Description
        number_of_instances (TYPE): Description
        plot_folder (TYPE): Description
        polyseme_name (TYPE): Description
        preposition (TYPE): Description
        preposition_models (TYPE): Description
        prototype (TYPE): Description
        prototype_csv (TYPE): Description
        rank (int): Description
        regression_weights_csv (TYPE): Description
        study_info (TYPE): Description
        train_scenes (TYPE): Description
        weights (TYPE): Description
    """

    def __init__(self, model_name, study_info_, preposition, polyseme_name, train_scenes, eq_feature_dict=None,
                 greater_feature_dict=None, less_feature_dict=None, features_to_remove=None, oversample: bool = False):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
            polyseme_name (TYPE): Description
            train_scenes (TYPE): Description
            eq_feature_dict (None, optional): Description
            greater_feature_dict (None, optional): Description
            less_feature_dict (None, optional): Description

            :param model_name:
            :param study_info_:
        """

        self.model_name = model_name
        self.study_info = study_info_
        self.polyseme_name = polyseme_name
        self.preposition = preposition
        self.train_scenes = train_scenes
        self.features_to_remove = features_to_remove
        self.oversample = oversample

        # Dictionary containing distinguishing features and their values
        self.eq_feature_dict = eq_feature_dict
        self.greater_feature_dict = greater_feature_dict
        self.less_feature_dict = less_feature_dict

        self.annotation_csv = POLYSEMY_MODEL_PROPERTY_FOLDER + self.model_name + '/annotations/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        # self.prototype_csv = self.study_info.polyseme_data_folder + self.model_name + '/prototypes/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        # self.mean_csv = self.study_info.polyseme_data_folder + self.model_name + '/means/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        # self.regression_weights_csv = self.study_info.polyseme_data_folder + self.model_name + '/regression weights/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.plot_folder = POLYSEMY_MODEL_PROPERTY_FOLDER + self.model_name + '/plots/'

        self.preposition_models = GeneratePrepositionModelParameters(self.study_info, self.preposition,
                                                                     self.train_scenes,
                                                                     features_to_remove=self.features_to_remove,
                                                                     polyseme=self, oversample=self.oversample)

        self.preposition_models.work_out_prototype_model()

        # Assign a rank/hierarchy to polysemes

        self.rank = self.get_rank()

        # Number of configurations fitting polysemes which were labelled as preposition by any participant
        self.number_of_instances = len(self.preposition_models.aff_dataset.index)

        self.weights = self.preposition_models.regression_weights
        self.prototype = self.preposition_models.prototype

    def potential_instance(self, value_array: np.ndarray):
        """Summary
        Checks if configuration could be a possible polyseme instance.
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # boolean checks whether the configuration could be an instance

        if len(value_array) != len(self.study_info.all_feature_keys):
            print(value_array)
            print(len(value_array))
            print(self.study_info.all_feature_keys)
            print(len(self.study_info.all_feature_keys))
            raise ValueError

        if self.eq_feature_dict is not None:
            for feature in self.eq_feature_dict:
                value = round(value_array[self.study_info.all_feature_keys.index(feature)], 6)
                condition = round(self.eq_feature_dict[feature], 6)

                if value != condition:
                    return False

        if self.greater_feature_dict is not None:
            for feature in self.greater_feature_dict:

                if value_array[self.study_info.all_feature_keys.index(feature)] < self.greater_feature_dict[feature]:
                    return False
        if self.less_feature_dict is not None:
            for feature in self.less_feature_dict:
                if value_array[self.study_info.all_feature_keys.index(feature)] > self.less_feature_dict[feature]:
                    return False
        return True

    def get_rank(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        ratio_feature_name = self.preposition_models.ratio_feature_name

        mean = self.preposition_models.train_possible_instances_dataset.mean(axis=0)[ratio_feature_name]

        self.rank = mean
        if np.isnan(self.rank):
            self.rank = 0

        return self.rank

    def plot(self, base_folder=None):
        """Summary
        """
        self.preposition_models.plot_models(base_folder)

    # def output_prototype_weight(self):
    #     """Summary
    #     """
    #     pf = pd.DataFrame(self.prototype, self.study_info.all_feature_keys)
    #
    #     pf.to_csv(self.prototype_csv)
    #
    #     wf = pd.DataFrame(self.weights, self.study_info.all_feature_keys)
    #
    #     wf.to_csv(self.regression_weights_csv)

    def output_definition(self, output_file=None):
        """Summary
        """
        if output_file is None:
            output_file = POLYSEMY_MODEL_PROPERTY_FOLDER + self.model_name + '/definitions/' + self.preposition + "-" + self.polyseme_name + ".csv"

        out = dict()
        out["eq_feature_dict"] = []
        out["greater_feature_dict"] = []
        out["less_feature_dict"] = []
        for feature in self.study_info.all_feature_keys:

            if self.eq_feature_dict != None:
                if feature in self.eq_feature_dict:

                    out["eq_feature_dict"].append(round(self.eq_feature_dict[feature], 6))
                else:
                    out["eq_feature_dict"].append("None")
            else:
                out["eq_feature_dict"].append("None")

            if self.greater_feature_dict != None:
                if feature in self.greater_feature_dict:

                    out["greater_feature_dict"].append(round(self.greater_feature_dict[feature], 6))
                else:
                    out["greater_feature_dict"].append("None")
            else:
                out["greater_feature_dict"].append("None")

            if self.less_feature_dict != None:
                if feature in self.less_feature_dict:

                    out["less_feature_dict"].append(round(self.less_feature_dict[feature], 6))
                else:
                    out["less_feature_dict"].append("None")
            else:
                out["less_feature_dict"].append("None")

        wf = pd.DataFrame(out, self.study_info.all_feature_keys)  # ["equality", "greater than", "less than"])

        wf.to_csv(output_file)


class PolysemyModel(Model):
    """Summary
    
    Attributes:
        test_prepositions (TYPE): Description
    """

    # Puts together preposition models and has various functions for testing
    def __init__(self, name, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST):
        """Summary
        
        Args:
            name (TYPE): Description
            test_scenes (TYPE): Description
            study_info_ (TYPE): Description
            :param test_prepositions:


        """

        Model.__init__(self, name, test_scenes, study_info_, test_prepositions=test_prepositions)

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        print("This shouldn't be called")


class DistinctPrototypePolysemyModel(PolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST,
                 preserve_empty_polysemes=False, baseline_model=None, features_to_remove=None,
                 oversample: bool = False):

        PolysemyModel.__init__(self, name, test_scenes, study_info_, test_prepositions=test_prepositions)

        self.oversample = oversample
        self.baseline_model = baseline_model
        self.train_scenes = train_scenes
        self.features_to_remove = features_to_remove
        # When empty polysemes are preserved their values are generated as normal
        # e.g. rank,number of instances  = 0. THis is useful for outputting data on the polysemes
        # When empty polysemes are not preserved, empty polysemes are assigned values from the baseline model.
        # I.e. Assign True when generating polysemes to explore data
        # Assign False when testing model.

        self.preserve_empty_polysemes = preserve_empty_polysemes
        # Dictionary of polysemes for each preposition
        # Non-shared polysemes don't share the prototype and this is the default
        self.polyseme_dict = dict()
        self.polyseme_dict = self.get_non_shared_prototype_polyseme_dict()

    def get_shared_prototype_polyseme_dict(self, old_dict):
        """Summary
        Gets polyseme dictionary from existing dictionary but makes each polyseme share the prototype.
        Returns:
            TYPE: Description
        """
        out = dict()

        for preposition in old_dict:
            out[preposition] = []
            for polyseme in old_dict[preposition]:
                new_pol = copy.deepcopy(polyseme)
                new_pol.prototype = self.baseline_model.preposition_model_dict[preposition].prototype

                out[preposition].append(new_pol)

        return out

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning and outputs a list of polysemes.
        :param preposition:
        :param original_salient_features:
        :return:
        """
        new_polysemes = self.generate_polysemes(preposition, original_salient_features)
        return new_polysemes

    def generate_polysemes(self, preposition, salient_features, train_scenes=None):
        """
        Generates polysemes based on ideal meaning discussion.
        Uses salient features and their threshold values.

        :param preposition:
        :param salient_features:
        :param train_scenes:
        :return:
        """

        if train_scenes is None:
            train_scenes = self.train_scenes

        if preposition in self.test_prepositions:
            polysemes = []

            g_dict = dict()
            l_dict = dict()

            for f in salient_features:

                if f.gorl == "l":
                    l_dict[f.name] = f.value
                else:
                    g_dict[f.name] = f.value

            # Canon

            p1 = Polyseme(self.name, self.study_info, preposition, "canon", train_scenes, greater_feature_dict=g_dict,
                          less_feature_dict=l_dict, features_to_remove=self.features_to_remove,
                          oversample=self.oversample)
            polysemes.append(p1)

            # Nearly canon
            x = len(salient_features) - 1
            while x >= 0:

                name_count = 0
                for pair in list(itertools.combinations(salient_features, x)):
                    name_count += 1
                    g_feature_dict = dict()
                    l_feature_dict = dict()

                    for f in salient_features:

                        if f not in pair:

                            if f.gorl == "l":
                                g_feature_dict[f.name] = f.value
                            else:
                                l_feature_dict[f.name] = f.value
                        if f in pair:

                            if f.gorl == "l":
                                l_feature_dict[f.name] = f.value
                            else:
                                g_feature_dict[f.name] = f.value
                    if x == 0:
                        p_name = "far" + str(name_count)
                    elif x == len(salient_features) - 1:
                        p_name = "near" + str(name_count)
                    else:
                        p_name = "not far" + str(name_count)
                    ply = Polyseme(self.name, self.study_info, preposition, p_name, train_scenes,
                                   greater_feature_dict=g_feature_dict, less_feature_dict=l_feature_dict,
                                   features_to_remove=self.features_to_remove, oversample=self.oversample)
                    polysemes.append(ply)
                x = x - 1

            if self.preserve_empty_polysemes:
                pass
            else:

                for poly in polysemes:

                    if poly.number_of_instances == 0:
                        # In the case there are no training instances (rank=0)
                        # Set the general parameters

                        poly.weights = self.baseline_model.preposition_model_dict[preposition].regression_weights
                        poly.prototype = self.baseline_model.preposition_model_dict[preposition].prototype

                        ratio_feature_name = GeneratePrepositionModelParameters.ratio_feature_name

                        poly.rank = self.baseline_model.preposition_model_dict[preposition].aff_dataset.mean(axis=0)[
                            ratio_feature_name]

                        if np.isnan(poly.rank):
                            poly.rank = 0

            polyseme_list = self.modify_polysemes(polysemes)
            return polyseme_list
        else:
            return []

    def modify_polysemes(self, polyseme_list):
        """ This method is overidden by shared model which modifies the polysemes to share the prototype"""
        return polyseme_list

    def get_non_shared_prototype_polyseme_dict(self):
        """Summary

        Returns:
            TYPE: Description
        """
        out = dict()

        contact03 = self.feature_processer.convert_normal_value_to_standardised("contact_proportion", 0.3)

        above09 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.9)
        above07 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.7)

        sup09 = self.feature_processer.convert_normal_value_to_standardised("support", 0.9)
        b07 = self.feature_processer.convert_normal_value_to_standardised("bbox_overlap_proportion", 0.7)
        lc075 = self.feature_processer.convert_normal_value_to_standardised("location_control", 0.75)
        lc025 = self.feature_processer.convert_normal_value_to_standardised("location_control", 0.25)
        gf09 = self.feature_processer.convert_normal_value_to_standardised("g_covers_f", 0.9)
        bl09 = self.feature_processer.convert_normal_value_to_standardised("below_proportion", 0.9)
        fg09 = self.feature_processer.convert_normal_value_to_standardised("f_covers_g", 0.9)

        hd01 = self.feature_processer.convert_normal_value_to_standardised("horizontal_distance", 0.1)

        # On

        f1 = SalientFeature("above_proportion", above09, "g")
        f2 = SalientFeature("support", sup09, "g")
        f3 = SalientFeature("contact_proportion", contact03, "g")
        on_salient_features = [f1, f2, f3]

        out["on"] = self.refine_ideal_meaning("on", on_salient_features)

        # In
        f1 = SalientFeature("bbox_overlap_proportion", b07, "g")
        f2 = SalientFeature("location_control", lc075, "g")

        in_salient_features = [f1, f2]

        out["in"] = self.refine_ideal_meaning("in", in_salient_features)

        # Under
        f1 = SalientFeature("g_covers_f", gf09, "g")
        f2 = SalientFeature("below_proportion", bl09, "g")

        under_salient_features = [f1, f2]

        out["under"] = self.refine_ideal_meaning("under", under_salient_features)

        # Over
        f1 = SalientFeature("f_covers_g", fg09, "g")
        f2 = SalientFeature("above_proportion", above07, "g")

        over_salient_features = [f1, f2]

        out["over"] = self.refine_ideal_meaning("over", over_salient_features)

        # on top of

        f1 = SalientFeature("above_proportion", above09, "g")
        f3 = SalientFeature("contact_proportion", contact03, "g")
        ontopof_salient_features = [f1, f3]
        out["on top of"] = self.refine_ideal_meaning("on top of", ontopof_salient_features)

        # inside
        f1 = SalientFeature("bbox_overlap_proportion", b07, "g")

        inside_salient_features = [f1]

        out["inside"] = self.refine_ideal_meaning("inside", inside_salient_features)
        # below
        f1 = SalientFeature("horizontal_distance", hd01, "l")
        f2 = SalientFeature("below_proportion", bl09, "g")

        below_salient_features = [f1, f2]

        out["below"] = self.refine_ideal_meaning("below", below_salient_features)

        # above
        f1 = SalientFeature("horizontal_distance", hd01, "l")
        f2 = SalientFeature("above_proportion", above07, "g")

        above_salient_features = [f1, f2]

        out["above"] = self.refine_ideal_meaning("above", above_salient_features)

        # against

        f1 = SalientFeature("horizontal_distance", hd01, "l")
        f2 = SalientFeature("location_control", lc025, "g")
        f3 = SalientFeature("contact_proportion", contact03, "g")

        against_salient_features = [f1, f2, f3]

        out["against"] = self.refine_ideal_meaning("against", against_salient_features)

        return out

    def get_possible_polysemes(self, preposition, value_array):
        """Summary
        Returns a list of possible polysemes for the given configuration.
        Args:
            preposition (TYPE): Description
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description

        Returns:
            TYPE: Description
        """
        out = []

        for polyseme in self.polyseme_dict[preposition]:
            if polyseme.potential_instance(value_array):
                out.append(polyseme)
        return out

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        '''
        Finds similarity to possible polysemes and multiplies by polyseme rank.
        May be better to pass configuration as parameter rather than scene, figure, ground names


        :param study:
        :param preposition:
        :param scene:
        :param figure:
        :param ground:
        :return:
        '''
        out = 0
        pps = self.get_possible_polysemes(preposition, value_array)
        if len(pps) == 0:
            print(self.name)
            print(preposition)
            print(scene)
            print(figure)
            print(ground)
            raise ValueError("Error: No polyseme given for:")

        for polyseme in pps:

            prototype_array = polyseme.prototype
            weight_array = polyseme.weights
            new = SemanticMethods.semantic_similarity(weight_array, value_array, prototype_array)

            new = new * polyseme.rank

            if new > out:
                out = new

        return out

    def get_datafolder_csv(self, preposition, data_folder):
        """
        Gets string for csv file when outputting info
        :param preposition:
        :return:
        """

        return POLYSEMY_MODEL_PROPERTY_FOLDER + self.name + '/' + data_folder + '/' + preposition + " -" + data_folder + ".csv"

    def output_polyseme_info(self, base_folder=None):
        """Summary
        Outputs polyseme info from model.
        """
        d = self.polyseme_dict
        if base_folder is None:
            base_folder = ""

        for preposition in d:
            rank_out = dict()
            prototype_out = dict()
            weight_out = dict()
            mean_out = dict()
            print(("Outputting:" + preposition))
            for polyseme in d[preposition]:
                # polyseme.output_prototype_weight()
                polyseme.output_definition(
                    base_folder + POLYSEMY_MODEL_PROPERTY_FOLDER + self.name + '/definitions/' + preposition + "-" + polyseme.polyseme_name + ".csv"
                )
                polyseme.plot(base_folder=base_folder)

                polyseme.preposition_models.aff_dataset.to_csv(base_folder + polyseme.annotation_csv)

                rank_out[preposition + "-" + polyseme.polyseme_name] = [
                    len(polyseme.preposition_models.aff_dataset.index),
                    polyseme.rank]

                prototype_out[preposition + "-" + polyseme.polyseme_name] = polyseme.prototype
                weight_out[preposition + "-" + polyseme.polyseme_name] = polyseme.weights
                mean_out[preposition + "-" + polyseme.polyseme_name] = polyseme.preposition_models.affFeatures.mean()

            number_df = pd.DataFrame(rank_out, ["Number", "Rank"])
            number_df.to_csv(base_folder + self.get_datafolder_csv(preposition, "ranks"))

            prototype_df = pd.DataFrame(prototype_out, self.study_info.all_feature_keys)
            prototype_df.to_csv(base_folder + self.get_datafolder_csv(preposition, "prototypes"))
            weight_df = pd.DataFrame(weight_out, self.study_info.all_feature_keys)
            weight_df.to_csv(base_folder + self.get_datafolder_csv(preposition, "regression weights"))
            mean_df = pd.DataFrame(mean_out, self.study_info.all_feature_keys)
            mean_df.to_csv(base_folder + self.get_datafolder_csv(preposition, "means"))


class DistinctPrototypeSupervisedPolysemyModel(DistinctPrototypePolysemyModel):
    # I think the differnce from above is that this isn't oversampled.
    # And it uses all scenes as train and validation.
    def __init__(self, name, train_scenes, test_scenes, study_info_, baseline_model: PrototypeModel,
                 test_prepositions=PREPOSITION_LIST,
                 preserve_empty_polysemes=False,
                 features_to_remove=None):

        DistinctPrototypePolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove,
                                                oversample=False)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by testing a validation set.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """
        if preposition in self.test_prepositions:
            new_salient_features = []

            # Each of the salient features are proportions so we use these values

            train_scenes = self.train_scenes
            validation_scenes = self.train_scenes

            # TODO: change this to test all combinations?
            stndardsd_values_to_try_dict = dict()

            for f in original_salient_features:
                g_values_to_try = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                l_values_to_try = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
                if f.name == "horizontal_distance":
                    g_values_to_try = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                    l_values_to_try = g_values_to_try
                if f.name == "contact_proportion":
                    g_values_to_try = l_values_to_try
                if f.gorl == "l":
                    values_to_try = l_values_to_try.copy()
                else:
                    values_to_try = g_values_to_try.copy()
                stndardsd_values_to_try_dict[f] = []
                [stndardsd_values_to_try_dict[f].append(
                    self.feature_processer.convert_normal_value_to_standardised(f.name, v))
                 for v
                 in values_to_try]

            for f in original_salient_features:

                best_value = None
                best_score = 0
                for v in stndardsd_values_to_try_dict[f]:

                    total = self.test_ideal_feature_value(train_scenes, validation_scenes, preposition,
                                                          original_salient_features, f.name, v)

                    if total > best_score:
                        best_score = total
                        best_value = v

                if best_value is None:
                    raise ValueError("best_value unassigned")

                # The original feature is updated, which is better for training the next feature
                f.value = best_value
                new_salient_features.append(f)
            new_polysemes = self.generate_polysemes(preposition, new_salient_features)
            return new_polysemes
        else:
            return None

    def test_ideal_feature_value(self, train_scenes, validation_scenes, preposition, original_salient_features, feature,
                                 value):
        """
        Generates new polysemes for the model from the given feature and value,
        then tests on given test scenes.
        :param train_scenes:
        :param preposition:
        :param original_salient_features:
        :param feature:
        :param value:
        :return:
        """

        # First update the salient features
        new_salient_features = []
        for f in original_salient_features:
            new_f = copy.deepcopy(f)
            if new_f.name == feature:
                new_f.value = value
            new_salient_features.append(new_f)

        # Create new polysemes
        new_polysemes = self.generate_polysemes(preposition, new_salient_features, train_scenes=train_scenes)
        # The polyseme dict is updated here so that the model score can be calculated

        self.polyseme_dict[preposition] = new_polysemes

        all_constraints = self.constraint_dict[preposition]
        # Constraints to test on
        test_constraints = []

        for c in all_constraints:
            if c.scene in validation_scenes:
                test_constraints.append(c)

        # Get score for preposition
        score_two = self.weighted_score(test_constraints)

        return score_two


class KMeansPolysemyModel(PolysemyModel):
    name = "KMeans Model"
    cluster_numbers = {'on': 8, 'in': 4, 'under': 4, 'over': 4, 'inside': 2, 'on top of': 4, 'below': 4, 'above': 4,
                       'against': 8}

    def __init__(self, preposition_model_dict, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST, ):
        PolysemyModel.__init__(self, KMeansPolysemyModel.name, test_scenes, study_info_,
                               test_prepositions=test_prepositions)

        self.preposition_model_dict = preposition_model_dict
        self.cluster_dict = self.get_cluster_dict()

    def get_cluster_dict(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Number of non-empty polysemes from polysemy model for all scenes
        # Actually none are empty, even for on

        out = dict()

        for preposition in self.test_prepositions:
            out[preposition] = []

            p_model_parameters = self.preposition_model_dict[preposition]

            # All selected instances
            possible_instances = p_model_parameters.affFeatures

            ratio_feature_name = p_model_parameters.ratio_feature_name
            sample_weights = p_model_parameters.aff_dataset[ratio_feature_name]

            # Issue that sometimes there's more samples than clusters
            # Set random state to make randomness deterministic for repeatability
            km = KMeans(
                n_clusters=self.cluster_numbers[preposition], random_state=0

            )
            km.fit(possible_instances, sample_weight=sample_weights)

            # Work out cluster ranks
            # Group configurations by their closest cluster centre.
            # THen find the average selectionratio for each group.

            weights_used_features = p_model_parameters.regression_weights_used_features

            cluster_ratio_sums = []
            cluster_number_of_instances = []
            for i in range(len(km.cluster_centers_)):
                cluster_ratio_sums.append(0)
                cluster_number_of_instances.append(0)

            for index, row in p_model_parameters.feature_dataframe.iterrows():
                # For each configuration add ratio to totals of closest centre

                # Note dropping columns from dataset preserves row order i.e.
                # row order of feature_dataframe = train_datset
                ratio_of_instance = p_model_parameters.train_dataset.at[index, ratio_feature_name]

                v = row.values
                # Convert values to np array
                v = np.array(v)

                sem_distance = -1
                chosen_centre = 0
                chosen_index = -1
                # Get closest centre
                for i in range(len(km.cluster_centers_)):

                    centre = km.cluster_centers_[i]

                    distance = SemanticMethods.semantic_distance(weights_used_features, v, centre)

                    if sem_distance == -1:
                        sem_distance = distance
                        chosen_centre = centre
                        chosen_index = i
                    elif distance < sem_distance:
                        sem_distance = distance
                        chosen_centre = centre
                        chosen_index = i
                # Update sums

                cluster_ratio_sums[chosen_index] += ratio_of_instance
                cluster_number_of_instances[chosen_index] += 1

            # Add clusters to dictionary.
            for i in range(len(km.cluster_centers_)):
                if cluster_number_of_instances[i] != 0:
                    rank = cluster_ratio_sums[i] / cluster_number_of_instances[i]
                else:
                    rank = 0

                new_c = ClusterInModel(preposition, km.cluster_centers_[i], weights_used_features, rank)
                out[preposition].append(new_c)

        return out

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        """Summary
        # Finds most similar cluster centre to point. Multiplies similarity to that cluster by cluster rank
        Args:
            preposition (TYPE): Description
            point (TYPE): Description

        Returns:
            TYPE: Description
            :param study:
            :param study:
        """

        clusters = self.cluster_dict[preposition]
        # Weight array uses weights assigned to baseline model
        # Same weights for all clusters for given preposition
        weight_array = clusters[0].weights
        closest_centre_typicality = 0
        closest_cluster = 0
        # Unused features must be removed here as weight and prototype array don't account for them.
        new_point = self.preposition_model_dict[preposition].remove_unused_features_from_array(value_array)
        for cluster in clusters:
            prototype_array = cluster.centre

            new = SemanticMethods.semantic_similarity(weight_array, new_point, prototype_array)
            if new > closest_centre_typicality:
                closest_centre_typicality = new
                closest_cluster = cluster

        out = closest_centre_typicality * closest_cluster.rank

        return out

    def folds_check(self, folds):
        for f in folds:
            # And also check that there are enough training samples for the K-Means model
            # in scenes not in fold
            # (samples must be greater than number of clusters..)
            scenes_not_in_fold = []
            for sc in self.study_info.scene_name_list:
                if sc not in f:
                    scenes_not_in_fold.append(sc)
            for preposition in self.test_prepositions:
                # Add some features to remove to ignore print out
                prep_model = GeneratePrepositionModelParameters(self.study_info, preposition, scenes_not_in_fold,
                                                                features_to_remove=Configuration.object_specific_features)
                if len(prep_model.affFeatures.index) < KMeansPolysemyModel.cluster_numbers[preposition]:
                    return False


class GeneratePolysemeModels(ModelGenerator):
    """Summary
    
    Attributes:
        baseline_model (TYPE): Description
        cluster_model (TYPE): Description
        models (TYPE): Description
        non_shared (TYPE): Description
        preserve_empty_polysemes (TYPE): Description
        shared (TYPE): Description
        study_info (TYPE): Description
        test_scenes (TYPE): Description
        train_scenes (TYPE): Description
    """
    # main model we are testing
    # name of the model we want to compare with other models, and use to test particular features

    # refined_distinct_model_name = "Refined Distinct Model"
    distinct_model_name = "Distinct Prototype"
    shared_model_name = "Shared Prototype"

    distinct_supervised_model_name = "Supervised Distinct Model"
    # shared_refined_model_name = "Refined Shared Model"

    # distinct_median_model_name = "Median Distinct Model"
    # shared_median_model_name = "Median Shared Model"

    baseline_model_name = "Baseline Model"
    cluster_model_name = KMeansPolysemyModel.name

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=None,
                 preserve_empty_polysemes=False):
        """Summary
        
        Args:
            train_scenes (TYPE): Description
            test_scenes (TYPE): Description
            study_info_ (TYPE): Description
            preserve_empty_polysemes (bool, optional): Description
            :param test_prepositions:
            :param study_info_:
        
        Deleted Parameters:
            constraint_dict (None, optional): Description
        """

        ModelGenerator.__init__(self, train_scenes, test_scenes, study_info_, test_prepositions)

        # When empty polysemes are preserved their values are generated as normal
        # e.g. rank,numebr of instances  = 0. THis is useful for outputting data on the polysemes
        # When empty polysemes are not preserved, empty polysemes are assigned values from the baseline model.

        self.preserve_empty_polysemes = preserve_empty_polysemes

        # First generate baseline model
        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in self.test_prepositions:
            M = GeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
                                                   features_to_remove=self.features_to_remove)
            M.work_out_prototype_model()
            preposition_models_dict[p] = M

        self.preposition_parameters_dict = preposition_models_dict
        self.baseline_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info,
                                             test_prepositions=self.test_prepositions)
        # Update some attributes
        self.baseline_model.name = self.baseline_model_name
        self.baseline_model.unsatisfied_constraints_csv = self.baseline_model.study_info.name + "/polysemy/unsatisfied constraints/" + self.baseline_model.name + ".csv"

        self.neural_categorisation = NeuralNetworkCategorisationModel(self.preposition_parameters_dict,
                                                                      self.test_scenes,
                                                                      self.study_info,
                                                                      test_prepositions=self.test_prepositions)

        # self.cluster_model = KMeansPolysemyModel(self.preposition_parameters_dict, self.test_scenes, self.study_info,
        #                                          test_prepositions=self.test_prepositions)

        # self.non_shared = DistinctPrototypePolysemyModel(GeneratePolysemeModels.distinct_model_name, self.train_scenes,
        #                                                  self.test_scenes, self.study_info,
        #                                                  test_prepositions=self.test_prepositions,
        #                                                  preserve_empty_polysemes=self.preserve_empty_polysemes,
        #                                                  baseline_model=self.baseline_model,
        #                                                  features_to_remove=self.features_to_remove)

        self.distinct_supervised_model = DistinctPrototypeSupervisedPolysemyModel(self.distinct_supervised_model_name,
                                                                                  train_scenes, test_scenes,
                                                                                  self.study_info,
                                                                                  self.baseline_model,
                                                                                  test_prepositions=self.test_prepositions,
                                                                                  features_to_remove=self.features_to_remove)

        # # To avoid repeating computations make a copy of non-shared and edit attributes.
        # self.shared = copy.deepcopy(self.non_shared)
        # self.shared.name = GeneratePolysemeModels.shared_model_name
        # self.shared.polyseme_dict = self.shared.get_shared_prototype_polyseme_dict(self.shared.polyseme_dict)

        self.generate_model_lists()


def test_models():
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """

    compare_models(10, 10, GeneratePolysemeModels, POLYSEMY_SCORES_FOLDER, test_prepositions=POLYSEMOUS_PREPOSITIONS)


def test_all_prepositions():
    compare_models(10, 10, GeneratePolysemeModels, ALL_PREPS_POLYSEMY_SCORES_FOLDER, test_prepositions=PREPOSITION_LIST)


def output_typicality(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("outputting typicalities")

    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_)
    p_models = generated_polyseme_models.models
    for model in p_models:

        for preposition in PREPOSITION_LIST:
            model.output_typicalities(preposition)


def output_all_polyseme_info():
    """Summary
    :param study_info_:

    Args:
        study_info_ (TYPE): Description
    """
    print("outputting all polyseme info")
    study_info_ = StudyInfo("2019 study")
    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_,
                                                       preserve_empty_polysemes=True)
    generated_polyseme_models.non_shared.output_polyseme_info()
    generated_polyseme_models.distinct_supervised_model.output_polyseme_info()


def main():
    """Un/comment functions to run tests and outputs
    
    Args:
        study_info_ (TYPE): Description
        :param study_info_:
    
    Deleted Parameters:
        constraint_dict (TYPE): Description
    """
    # study_info = StudyInfo("2019 study")
    # output_all_polyseme_info(study_info_)
    # Clustering

    # Polysemes and performance

    # output_typicality(study_info_)
    output_all_polyseme_info()
    # test_models()
    # test_all_prepositions()


if __name__ == '__main__':
    main()

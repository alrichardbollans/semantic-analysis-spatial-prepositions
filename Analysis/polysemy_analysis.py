"""Summary
First run compile_instances.py
Attributes:
    comp_filetag (str): Description
    non_polysemous_prepositions (list): Description
    polysemous_preposition_list (list): Description
    preposition_list (TYPE): Description
    sv_filetag (str): Description

"""
#

# Standard imports
import copy
import pandas as pd
import numpy as np
import itertools

# Ml modules

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Modules for plotting
import matplotlib as mpl

from basic_model_testing import TestModels, GeneratePrepositionModelParameters, Model, MultipleRuns, \
    SemanticMethods, PrototypeModel
from data_import import Configuration, StudyInfo
from compile_instances import SemanticCollection, ComparativeCollection

# Useful global variables
sv_filetag = SemanticCollection.filetag  # Tag for sv task files
comp_filetag = ComparativeCollection.filetag  # Tag for comp task files
preposition_list = StudyInfo.preposition_list
polysemous_preposition_list = preposition_list  # ['in', 'on', 'under', 'over']  # list of prepositions which exist in the data
non_polysemous_prepositions = ["inside", "above", "below", "on top of", 'against']


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
                 greater_feature_dict=None, less_feature_dict=None, features_to_remove=None):
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

        # Dictionary containing distinguishing features and their values
        self.eq_feature_dict = eq_feature_dict
        self.greater_feature_dict = greater_feature_dict
        self.less_feature_dict = less_feature_dict

        self.annotation_csv = self.study_info.polyseme_data_folder + self.model_name + '/annotations/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.prototype_csv = self.study_info.polyseme_data_folder + self.model_name + '/prototypes/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.mean_csv = self.study_info.polyseme_data_folder + self.model_name + '/means/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.regression_weights_csv = self.study_info.polyseme_data_folder + self.model_name + '/regression weights/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.plot_folder = self.study_info.polyseme_data_folder + self.model_name + '/plots/'

        self.preposition_models = GeneratePrepositionModelParameters(self.study_info, self.preposition,
                                                                     self.train_scenes,
                                                                     features_to_remove=self.features_to_remove,
                                                                     polyseme=self)

        # Assign a rank/hierarchy to polysemes

        self.rank = self.get_rank()
        # Number of configurations fitting polysemes which were labelled as preposition by any participant
        self.number_of_instances = self.get_number_of_instances()

        self.weights = self.preposition_models.regression_weights
        self.prototype = self.preposition_models.prototype

    def potential_instance(self, scene, figure, ground):
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

        r = Configuration(scene, figure, ground, self.study_info)

        r.load_from_csv()
        if self.eq_feature_dict is not None:
            for feature in self.eq_feature_dict:
                value = round(r.set_of_features[feature], 6)
                condition = round(self.eq_feature_dict[feature], 6)
                if value != condition:
                    return False

        if self.greater_feature_dict is not None:
            for feature in self.greater_feature_dict:

                if r.set_of_features[feature] < self.greater_feature_dict[feature]:
                    return False
        if self.less_feature_dict is not None:
            for feature in self.less_feature_dict:
                if r.set_of_features[feature] > self.less_feature_dict[feature]:
                    return False
        return True

    def get_number_of_instances(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return len(self.preposition_models.aff_dataset.index)

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

    def plot(self):
        """Summary
        """
        self.preposition_models.plot_models()

    def output_prototype_weight(self):
        """Summary
        """
        pf = pd.DataFrame(self.prototype, self.study_info.all_feature_keys)

        pf.to_csv(self.prototype_csv)

        wf = pd.DataFrame(self.weights, self.study_info.all_feature_keys)

        wf.to_csv(self.regression_weights_csv)

    def output_definition(self):
        """Summary
        """
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

        wf.to_csv(
            self.study_info.polyseme_data_folder + self.model_name + '/definitions/' + self.preposition + "-" + self.polyseme_name + ".csv")


class PolysemyModel(Model):
    """Summary
    
    Attributes:
        test_prepositions (TYPE): Description
    """

    # Puts together preposition models and has various functions for testing
    def __init__(self, name, test_scenes, study_info_):
        """Summary
        
        Args:
            name (TYPE): Description
            test_scenes (TYPE): Description
            study_info_ (TYPE): Description
        

        """
        print("generating model:" + name)
        print("Number of test scenes:" + str(len(test_scenes)))
        Model.__init__(self, name, test_scenes, study_info_)
        self.test_prepositions = polysemous_preposition_list

    def get_typicality(self, preposition, point, scene=None, figure=None, ground=None):
        print("This shouldn't be called")


class DistinctPrototypePolysemyModel(PolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):

        PolysemyModel.__init__(self, name, test_scenes, study_info_)
        self.baseline_model = baseline_model
        self.train_scenes = train_scenes
        self.features_to_remove = features_to_remove
        # When empty polysemes are preserved their values are generated as normal
        # e.g. rank,numebr of instances  = 0. THis is useful for outputting data on the polysemes
        # When empty polysemes are not preserved, empty polysemes are assigned values from the baseline model.

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

        if preposition in polysemous_preposition_list:
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
                          less_feature_dict=l_dict, features_to_remove=self.features_to_remove)
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
                                   features_to_remove=self.features_to_remove)
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

    def get_possible_polysemes(self, preposition, scene, figure, ground):
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
            if polyseme.potential_instance(scene, figure, ground):
                out.append(polyseme)
        return out

    def get_typicality(self, preposition, point, scene=None, figure=None, ground=None):
        '''
        Finds similarity to possible polysemes and multiplies by polyseme rank.
        :param preposition:
        :param point:
        :param scene:
        :param figure:
        :param ground:
        :return:
        '''
        out = 0
        pps = self.get_possible_polysemes(preposition, scene, figure, ground)
        if len(pps) == 0:
            print(preposition)
            print(scene)
            print(figure)
            print(ground)
            raise ValueError("Error: No polyseme given for:")

        for polyseme in pps:

            prototype_array = polyseme.prototype
            weight_array = polyseme.weights
            new = SemanticMethods.semantic_similarity(weight_array, point, prototype_array)

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

        return self.study_info.polyseme_data_folder + self.name + '/' + data_folder + '/' + preposition + " -" + data_folder + ".csv"

    def output_polyseme_info(self):
        """Summary
        Outputs polyseme info from model.
        """
        d = self.polyseme_dict

        for preposition in d:
            rank_out = dict()
            prototype_out = dict()
            weight_out = dict()
            mean_out = dict()
            print(("Outputting:" + preposition))
            for polyseme in d[preposition]:
                # polyseme.output_prototype_weight()
                polyseme.output_definition()
                polyseme.plot()

                polyseme.preposition_models.aff_dataset.to_csv(polyseme.annotation_csv)
                # polyseme.preposition_models.affFeatures.mean().to_csv(polyseme.mean_csv, header=False)

                rank_out[preposition + "-" + polyseme.polyseme_name] = [polyseme.get_number_of_instances(),
                                                                        polyseme.rank]

                prototype_out[preposition + "-" + polyseme.polyseme_name] = polyseme.prototype
                weight_out[preposition + "-" + polyseme.polyseme_name] = polyseme.weights
                mean_out[preposition + "-" + polyseme.polyseme_name] = polyseme.preposition_models.affFeatures.mean()

            number_df = pd.DataFrame(rank_out, ["Number", "Rank"])
            number_df.to_csv(self.get_datafolder_csv(preposition, "ranks"))

            prototype_df = pd.DataFrame(prototype_out, self.study_info.all_feature_keys)
            prototype_df.to_csv(self.get_datafolder_csv(preposition, "prototypes"))
            weight_df = pd.DataFrame(weight_out, self.study_info.all_feature_keys)
            weight_df.to_csv(self.get_datafolder_csv(preposition, "regression weights"))
            mean_df = pd.DataFrame(mean_out, self.study_info.all_feature_keys)
            mean_df.to_csv(self.get_datafolder_csv(preposition, "means"))


class DistinctPrototypeMedianPolysemyModel(DistinctPrototypePolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
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

        new_salient_features = []
        for f in original_salient_features:
            new_f = copy.deepcopy(f)

            median = self.baseline_model.preposition_model_dict[preposition].goodAllFeatures[new_f.name].median()
            new_f.value = median

            new_salient_features.append(new_f)

        new_polysemes = self.generate_polysemes(preposition, new_salient_features)
        return new_polysemes


class DistinctPrototypeRefinedPolysemyModel(DistinctPrototypePolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by testing a validation test.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """
        new_salient_features = []
        if preposition in polysemous_preposition_list:
            # Each of the salient features are proportions so we use these values
            # This makes generation non-deterministic for these models
            train_scenes, validation_scenes = train_test_split(self.train_scenes, test_size=0.5)
            g_values_to_try = [0.5, 0.6, 0.7, 0.8, 0.9]
            l_values_to_try = [0.1, 0.2, 0.2, 0.4, 0.5]

            for f in original_salient_features:
                if f.name == "horizontal_distance":
                    g_values_to_try = [0.05, 0.1, 0.15, 0.2]
                    l_values_to_try = g_values_to_try
                if f.name == "contact_proportion":
                    g_values_to_try = [0.1, 0.2, 0.3, 0.4, 0.5]
                    l_values_to_try = [0.1, 0.2, 0.3, 0.4, 0.5]
                if f.gorl == "l":
                    values_to_try = l_values_to_try.copy()
                else:
                    values_to_try = g_values_to_try.copy()

                best_value = 0
                best_score = 0
                for v in values_to_try:
                    # Convert to standardised values
                    v = self.feature_processer.convert_normal_value_to_standardised(f.name, v)

                    score1 = self.test_ideal_feature_value(train_scenes, validation_scenes, preposition,
                                                           original_salient_features, f.name, v)
                    score2 = self.test_ideal_feature_value(validation_scenes, train_scenes, preposition,
                                                           original_salient_features, f.name, v)

                    total = score1 + score2
                    if total > best_score:
                        best_score = total
                        best_value = v

                if best_value == 0:
                    raise ValueError("best_value unassigned")

                # The original feature is updated, which is better for training the next feature
                f.value = best_value
                new_salient_features.append(f)
        new_polysemes = self.generate_polysemes(preposition, new_salient_features)
        return new_polysemes

    def test_ideal_feature_value(self, train_scenes, validation_scenes, preposition, original_salient_features, feature,
                                 value):
        """
        Generates new polysemes for the model from the given feature and value,
        then tests on given test scenes.
        :param train_scenes:
        :param test_scenes:
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

        allConstraints = self.constraint_dict[preposition]
        # Constraints to test on
        testConstraints = []

        for c in allConstraints:
            if c.scene in validation_scenes:
                testConstraints.append(c)

        # Get score for preposition
        score_two = self.weighted_score(testConstraints)

        return score_two


class SharedPrototypeRefinedPolysemyModel(DistinctPrototypeRefinedPolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypeRefinedPolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                       preserve_empty_polysemes=preserve_empty_polysemes,
                                                       baseline_model=baseline_model,
                                                       features_to_remove=features_to_remove)

    def modify_polysemes(self, polyseme_list):
        for poly in polyseme_list:
            poly.prototype = self.baseline_model.preposition_model_dict[poly.preposition].prototype
        return polyseme_list


class KMeansPolysemyModel(PolysemyModel):
    name = "KMeans Model"
    cluster_numbers = {'on': 8, 'in': 4, 'under': 4, 'over': 4, 'inside': 2, 'on top of': 4, 'below': 4, 'above': 4,
                       'against': 8}

    def __init__(self, preposition_model_dict, test_scenes, study_info_):
        PolysemyModel.__init__(self, KMeansPolysemyModel.name, test_scenes, study_info_)

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

        for preposition in polysemous_preposition_list:
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

    def get_typicality(self, preposition, point, scene=None, figure=None, ground=None):
        """Summary
        # Finds most similar cluster centre to point. Multiplies similarity to that cluster by cluster rank
        Args:
            preposition (TYPE): Description
            point (TYPE): Description

        Returns:
            TYPE: Description
        """

        clusters = self.cluster_dict[preposition]
        # Weight array uses weights assigned to baseline model
        # Same weights for all clusters for given preposition
        weight_array = clusters[0].weights
        closest_centre_typicality = 0
        closest_cluster = 0
        # Unused features must be removed here as weight and prototype array don't account for them.
        new_point = self.preposition_model_dict[preposition].remove_unused_features_from_array(point)
        for cluster in clusters:
            prototype_array = cluster.centre

            new = SemanticMethods.semantic_similarity(weight_array, new_point, prototype_array)
            if new > closest_centre_typicality:
                closest_centre_typicality = new
                closest_cluster = cluster

        out = closest_centre_typicality * closest_cluster.rank

        return out


class GeneratePolysemeModels:
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
    our_model_name = "Refined Distinct Model"

    distinct_model_name = "Distinct Prototype"
    shared_model_name = "Shared Prototype"

    distinct_refined_model_name = "Refined Distinct Model"
    shared_refined_model_name = "Refined Shared Model"

    distinct_median_model_name = "Median Distinct Model"
    shared_median_model_name = "Median Shared Model"

    baseline_model_name = "Baseline Model"
    cluster_model_name = KMeansPolysemyModel.name

    def __init__(self, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False):
        """Summary
        
        Args:
            train_scenes (TYPE): Description
            test_scenes (TYPE): Description
            study_info_ (TYPE): Description
            preserve_empty_polysemes (bool, optional): Description
            :param study_info_:
        
        Deleted Parameters:
            constraint_dict (None, optional): Description
        """

        self.study_info = study_info_
        # Scenes used to train models
        self.train_scenes = train_scenes
        # Scenes used to test models
        self.test_scenes = test_scenes
        self.features_to_remove = Configuration.ground_property_features.copy()
        # When empty polysemes are preserved their values are generated as normal
        # e.g. rank,numebr of instances  = 0. THis is useful for outputting data on the polysemes
        # When empty polysemes are not preserved, empty polysemes are assigned values from the baseline model.

        self.preserve_empty_polysemes = preserve_empty_polysemes

        # First generate baseline model
        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in preposition_list:
            M = GeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
                                                   features_to_remove=self.features_to_remove)
            preposition_models_dict[p] = M

        self.preposition_parameters_dict = preposition_models_dict
        self.baseline_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info)
        # Update some attributes
        self.baseline_model.test_prepositions = polysemous_preposition_list
        self.baseline_model.name = self.baseline_model_name
        self.baseline_model.unsatisfied_constraints_csv = self.baseline_model.study_info.name + "/polysemy/unsatisfied constraints/" + self.baseline_model.name + ".csv"

        self.cluster_model = KMeansPolysemyModel(self.preposition_parameters_dict, self.test_scenes, self.study_info)

        self.non_shared = DistinctPrototypePolysemyModel(GeneratePolysemeModels.distinct_model_name, self.train_scenes,
                                                         self.test_scenes, self.study_info,
                                                         preserve_empty_polysemes=self.preserve_empty_polysemes,
                                                         baseline_model=self.baseline_model,
                                                         features_to_remove=self.features_to_remove)
        # # To avoid repeating computations make a copy of non-shared and edit attributes.
        # self.shared = copy.deepcopy(self.non_shared)
        # self.shared.name = GeneratePolysemeModels.shared_model_name
        # self.shared.polyseme_dict = self.shared.get_shared_prototype_polyseme_dict(self.shared.polyseme_dict)

        # self.refined = DistinctPrototypeRefinedPolysemyModel(GeneratePolysemeModels.distinct_refined_model_name,
        #                                                      self.train_scenes,
        #                                                      self.test_scenes, self.study_info,
        #                                                      preserve_empty_polysemes=self.preserve_empty_polysemes,
        #                                                      baseline_model=self.baseline_model,
        #                                                      features_to_remove=self.features_to_remove)

        # self.shared_refined = SharedPrototypeRefinedPolysemyModel(GeneratePolysemeModels.shared_refined_model_name,
        #                                                           self.train_scenes,
        #                                                           self.test_scenes, self.study_info,
        #                                                           preserve_empty_polysemes=self.preserve_empty_polysemes,
        #                                                           baseline_model=self.baseline_model,
        #                                                           features_to_remove=self.features_to_remove)

        # self.median = DistinctPrototypeMedianPolysemyModel(GeneratePolysemeModels.distinct_median_model_name,
        #                                                    self.train_scenes,
        #                                                    self.test_scenes, self.study_info,
        #                                                    preserve_empty_polysemes=self.preserve_empty_polysemes,
        #                                                    baseline_model=self.baseline_model,
        #                                                    features_to_remove=self.features_to_remove)

        # self.shared_median = copy.deepcopy(self.median)
        # self.shared_median.name = GeneratePolysemeModels.shared_median_model_name
        # self.shared_median.polyseme_dict = self.shared_median.get_shared_prototype_polyseme_dict(
        #     self.shared_median.polyseme_dict)

        self.models = [self.non_shared, self.baseline_model, self.cluster_model]  # , self.refined,
        # self.median]

        # self.models = [self.non_shared, self.shared, self.baseline_model, self.cluster_model, self.refined,
        #                self.shared_refined, self.median, self.shared_median]

        self.model_name_list = []
        for m in self.models:
            self.model_name_list.append(m.name)


class MultipleRunsPolysemyModels(MultipleRuns):
    """Summary
    
    Attributes:
        all_csv (TYPE): Description
        all_plot (TYPE): Description
        average_csv (TYPE): Description
        average_plot_pdf (TYPE): Description
        average_plot_title (TYPE): Description
        comparison_csv (TYPE): Description
        file_tag (TYPE): Description
        km_comparison_csv (TYPE): Description
        scores_plots_folder (TYPE): Description
        scores_tables_folder (TYPE): Description
        study_info (TYPE): Description
    """

    def __init__(self, study_info_, number_runs=None, k=None, compare=None):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            number_runs (None, optional): Description
            k (None, optional): Description
            compare (None, optional): Description
        
        Deleted Parameters:
            study_info (TYPE): Description
            constraint_dict (TYPE): Description
            features_to_test (None, optional): Description
        """
        self.study_info = study_info_

        MultipleRuns.__init__(self, GeneratePolysemeModels, self.study_info, number_runs=number_runs, test_size=None,
                              k=k,
                              compare=compare, features_to_test=None)

        self.scores_tables_folder = self.study_info.polysemy_score_folder + "tables"
        self.scores_plots_folder = self.study_info.polysemy_score_folder + "plots"

        self.all_csv = self.study_info.polysemy_score_folder + "all_test.csv"
        self.all_plot = self.study_info.polysemy_score_folder + "ScoresUsingAllData.pdf"

        if self.k is not None:
            self.file_tag = str(self.k) + "fold"
            self.average_plot_title = "Scores Using Repeated K-Fold Validation. K = " + str(self.k) + " N = " + str(
                self.number_runs)

            self.average_plot_pdf = self.scores_plots_folder + "/average" + self.file_tag + ".pdf"
            self.average_csv = self.scores_tables_folder + "/averagemodel scores " + self.file_tag + ".csv"
            self.comparison_csv = self.scores_tables_folder + "/repeatedcomparisons " + self.file_tag + ".csv"
            self.km_comparison_csv = self.scores_tables_folder + "/km_repeatedcomparisons " + self.file_tag + ".csv"

    # Overides inherited method
    def output(self):
        """Summary
        """
        # Handle outputting here so we're not always outputting
        self.average_dataframe = self.dataframe_dict["all_features"]
        # Reorder columns for output

        new_column_order = self.Generate_Models_all_scenes.model_name_list
        reordered_df = self.average_dataframe[new_column_order]
        reordered_df.to_csv(self.average_csv)

        self.plot_dataframe_bar_chart(self.average_dataframe, self.average_plot_pdf, "Preposition", "Score",
                                      self.average_plot_title)
        if self.compare is not None:
            # Output to csv
            self.comparison_df.to_csv(self.comparison_csv)

            self.km_comparison_df.to_csv(self.km_comparison_csv)


def output_all_polyseme_info(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("outputting all polyseme info")
    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_,
                                                       preserve_empty_polysemes=True)
    generated_polyseme_models.non_shared.output_polyseme_info()
    # generated_polyseme_models.refined.output_polyseme_info()


def test_on_all_scenes(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("test on all scenes")

    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_)

    p_models = generated_polyseme_models.models

    t = TestModels(p_models, "all")
    all_dataframe = t.score_dataframe.copy()

    all_dataframe.to_csv(study_info_.polysemy_score_folder + "all_test.csv")
    print(all_dataframe)


def test_model(runs, k, study_info_):
    """Summary
    
    Args:
        runs (TYPE): Description
        k (TYPE): Description
        study_info_ (TYPE): Description
        :param study_info_:
    """
    m = MultipleRunsPolysemyModels(study_info_, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print((m.average_dataframe))


def test_models(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    mpl.rcParams['font.size'] = 40
    mpl.rcParams['legend.fontsize'] = 37
    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.labelsize'] = 'medium'
    mpl.rcParams['ytick.labelsize'] = 'small'

    test_on_all_scenes(study_info_)
    test_model(2, 10, study_info_)
    # test_model(10, 10, study_info_)


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

        for preposition in polysemous_preposition_list:
            model.output_typicalities(preposition)


def main(study_info_):
    """Un/comment functions to run tests and outputs
    
    Args:
        study_info_ (TYPE): Description
        :param study_info_:
    
    Deleted Parameters:
        constraint_dict (TYPE): Description
    """
    output_all_polyseme_info(study_info_)
    # Clustering

    # Polysemes and performance

    # output_typicality(study_info_)
    test_models(study_info_)


if __name__ == '__main__':
    study_info = StudyInfo("2019 study")

    main(study_info)

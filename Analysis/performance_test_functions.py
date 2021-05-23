import operator
import os
import time

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from Analysis.classes import Constraint

from Analysis.data_import import Configuration, StudyInfo

PREPOSITION_LIST = StudyInfo.preposition_list

MODEL_EVALUATION_FOLDER = "model evaluation/"
BASIC_MODEL_SCORES_FOLDER = "model evaluation/basic scores/"
POLYSEMY_SCORES_FOLDER = "model evaluation/polysemy/"
ALL_PREPS_POLYSEMY_SCORES_FOLDER = "model evaluation/polysemy - all prepositions/"


class Model:
    """Summary


    """

    # Puts together preposition models and has various functions for testing
    def __init__(self, name, test_scenes, study_info_, test_prepositions=PREPOSITION_LIST,
                 constraint_csv_removed_users=None):
        """Summary

        Args:
        :param test_prepositions:

        """
        print("generating model:" + name)
        print("Number of test scenes:" + str(len(test_scenes)))
        self.study_info = study_info_
        self.test_scenes = test_scenes
        self.name = name

        self.feature_processer = self.study_info.feature_processor

        self.all_feature_keys = self.study_info.all_feature_keys

        # Prepositions to test
        self.test_prepositions = test_prepositions

        # Dictionary containing constraints to satisfy
        if constraint_csv_removed_users is None:
            self.constraint_dict = Constraint.read_from_csv(self.study_info.constraint_csv)
        else:
            self.constraint_dict = Constraint.read_from_csv(constraint_csv_removed_users)
        # Csv to write unsatisfied constraints when testing/training on all scenes
        self.unsatisfied_constraints_csv = "extra thesis results/unsatisfied constraints/" + self.name + ".csv"

    def folds_check(self, folds):
        return True

    def generate_arrays(self, salient_features):
        """
        Creates prototype and weight arrays from dictionary of salient features specifiying their (human-readable) typical values.

        :param salient_features:
        :returns prototype_array, weight_array
        """
        pro_array = []
        weight_array = []
        for feature in self.all_feature_keys:
            if feature not in salient_features:
                pro_array.append(0)
                weight_array.append(0)
            else:
                x = self.feature_processer.convert_normal_value_to_standardised(feature, salient_features[feature])
                pro_array.append(x)
                weight_array.append(1)

        pro_array = np.array(pro_array)
        weight_array = np.array(weight_array)

        return pro_array, weight_array

    def get_typicality(self, preposition, value_array, scene=None, figure=None,
                       ground=None, study=None):
        """Gets typicality of configuration for model. scene, figure, ground parameters are given for models which
        need these. Study parameter given to allow checking of configurations from different studies"""
        print("This shouldn't be called")

    def get_typicality_lhs(self, constraint):

        return self.get_typicality(constraint.preposition, constraint.lhs_values, constraint.scene, constraint.f1,
                                   constraint.ground)

    def get_typicality_rhs(self, constraint):
        return self.get_typicality(constraint.preposition, constraint.rhs_values, constraint.scene, constraint.f2,
                                   constraint.ground)

    def get_test_constraints(self, preposition):
        allConstraints = self.constraint_dict[preposition]
        # Constraints to test on
        testConstraints = []

        for c in allConstraints:
            if c.scene in self.test_scenes:
                testConstraints.append(c)
        return testConstraints

    def get_score(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Calculates scores on all constraints for particular model
        # Prototype dict is a dictionary of prototypes (1D arrays) for each preposition

        scores = []  # Scores for each preposition to be averaged later
        weight_totals = []  # Total constraint weights for each preposition
        totals = []  # Total number of constraints for each preposition
        average_score = 0
        weighted_average_score = 0
        total_weight_counter = 0

        for preposition in self.test_prepositions:

            testConstraints = self.get_test_constraints(preposition)
            if len(testConstraints) == 0:
                print(preposition)
                raise ValueError
            # Constraint info
            weight_counter = 0
            counter = 0
            for c in testConstraints:
                if c.weight == 0:
                    raise ValueError
                weight_counter += c.weight
                counter += 1
            total_weight_counter += weight_counter
            weight_totals.append(weight_counter)
            totals.append(counter)

            # Get score for preposition
            score_two = self.weighted_score(testConstraints)

            weighted_average_score += score_two

            score = float(score_two) / weight_counter
            average_score += score

            # score =  round(score,3)
            scores.append(score)

        average_score = float(average_score) / len(self.test_prepositions)
        weighted_average_score = float(weighted_average_score) / total_weight_counter
        # average_score =  round(average_score,3)
        # weighted_average_score =  round(weighted_average_score,3)
        scores.append(average_score)
        scores.append(weighted_average_score)

        self.weight_totals = weight_totals
        self.totals = totals
        self.scores = scores

        return scores

    def weighted_score(self, constraints):
        """Summary

        Args:
            preposition (TYPE): Description
            Constraints (TYPE): Description

        Returns:
            TYPE: Description
        """

        # # Calculates how well W and P satisfy the constraints, accounting for constraint weight
        counter = 0
        #
        for c in constraints:
            lhs = self.get_typicality_lhs(c)
            rhs = self.get_typicality_rhs(c)
            if c.is_satisfied(lhs, rhs):
                counter += c.weight
            elif lhs == rhs:
                counter += float(c.weight) / 2

        return counter

    def output_typicalities(self, preposition, input_csv=None, study_info=None):
        """Summary

        Args:
            preposition (TYPE): Description
            :param study_info: Study to output typicalities for
        """
        # output_csv = base_polysemy_folder+ "config typicalities/"+self.name+"-typicality_test-"+preposition+".csv"
        if input_csv is None:
            input_csv = MODEL_EVALUATION_FOLDER + "config typicalities/typicality-" + preposition + ".csv"
        if study_info is None:
            study_info = self.study_info

        config_list = study_info.config_list

        new_csv = False

        try:
            # print(self.name)
            # print("try to read")
            in_df = pd.read_csv(input_csv, index_col=0)

        except Exception as e:
            in_df = pd.DataFrame(columns=['scene', 'figure', 'ground', self.name])
            # print("unsusccefully read")
            new_csv = True
        # else:
        # 	pass
        finally:
            # pass

            # print(in_df)

            df_columns = in_df.columns
            for c in config_list:

                # Typicality is calculated for each configuration
                # To check whether a configuration fits a particular polyseme we need to include
                value_array = np.array(c.row)
                typicality = self.get_typicality(preposition, value_array, scene=c.scene, figure=c.figure,
                                                 ground=c.ground, study=study_info)
                if new_csv:
                    in_df = in_df.append(
                        {'scene': c.scene, 'figure': c.figure, 'ground': c.ground, self.name: typicality},
                        ignore_index=True)
                else:
                    row_index_in_df = in_df[(in_df['scene'] == c.scene) & (in_df['figure'] == c.figure) & (
                            in_df['ground'] == c.ground)].index.tolist()

                    # if self.name in df_columns:

                    in_df.at[row_index_in_df[0], self.name] = typicality
                # else:
                # in_df[self.name] =
            # print(preposition)
            in_df.to_csv(input_csv)

    def output_unsatisfied_constraints(self):
        if len(self.test_scenes) == 67:
            # Clear the unsatisfied constraint csv file first
            f = open(self.unsatisfied_constraints_csv, "w+")
            f.close()
            unsatisfied_constraints = []

            for preposition in self.test_prepositions:

                testConstraints = self.get_test_constraints(preposition)

                for c in testConstraints:
                    lhs = self.get_typicality_lhs(c)
                    rhs = self.get_typicality_rhs(c)
                    if c.is_satisfied(lhs, rhs):
                        pass
                    else:
                        unsatisfied_constraints.append(c)
            # Output unsatisfied constraints if training/testing on all scenes
            # This may not output all unsatisfied constraints if constraint dict has been modified

            for c in unsatisfied_constraints:
                c.write_to_csv(self.unsatisfied_constraints_csv)
        else:
            raise ValueError('Not testing on all scenes')

    def remove_features_from_array(self, value_array, features_to_remove):
        value_array = value_array
        new_array = []
        if features_to_remove is not None:
            for feature in self.all_feature_keys:

                if feature not in features_to_remove:
                    new_array.append(
                        value_array[self.all_feature_keys.index(feature)])
            return new_array
        else:
            return value_array


class MultipleRuns:
    """Summary

    """

    # This class carries out multiple runs of model tests and outputs the results
    # Number of runs must be specified as well as k for repeated k-fold sampling
    def __init__(self, model_generator, scores_tables_folder, scores_plots_folder, study_info_,
                 number_runs=None, test_prepositions=None,
                 k=None, compare=None, features_to_test=None,model_to_test_name =None):
        """Summary

        Args:

        """

        self.study_info = study_info_
        if test_prepositions is not None:
            self.test_prepositions = test_prepositions
        else:
            self.test_prepositions = PREPOSITION_LIST

        self.model_generator = model_generator

        self.number_runs = number_runs
        self.k = k
        if self.k == 1:
            raise Exception("k must be greater than 1")
        self.compare = compare
        self.features_to_test = features_to_test
        #Model to remove features from to test
        self.model_to_test_name = model_to_test_name

        self.run_count = 0
        # Dictionary of dataframes giving scores. Indexed by removed features.
        # When no features being removed for testing purposes, index is "all_features"
        self.dataframe_dict = dict()

        self.scene_list = self.study_info.scene_name_list

        self.scores_tables_folder = scores_tables_folder
        self.scores_plots_folder = scores_plots_folder

        self.get_file_strings()

        self.Generate_Models_all_scenes = self.generate_models(self.scene_list, self.scene_list)
        ## Model names being tested. Gets from GenerateModels instance as models being tested depends on instance.
        self.model_name_list = self.Generate_Models_all_scenes.model_name_list
        self.constraint_dict = self.Generate_Models_all_scenes.models[0].constraint_dict

        self.prepare_comparison_dicts()
        # overall_folds_dict contains overall scores on each fold for each model
        # our_model_feature_folds_dict contains scores for each preposition when feature is included
        # our_model_without_feature_folds_dict contains scores for each preposition when feature is removed
        self.overall_folds_dict, self.our_model_feature_folds_dict, self.our_model_without_feature_folds_dict = self.prepare_folds_dict()
        # Also include avg scores from folds
        self.avg_folds_dict = self.overall_folds_dict.copy()

        # following lists help confirm all scenes get used for both training and testing
        self.scenes_used_for_testing = []
        self.scenes_used_for_training = []

    def get_file_strings(self):
        if not os.path.isdir(self.scores_tables_folder):
            raise Exception(f"Not a valid path! {self.scores_tables_folder}")
        if not os.path.isdir(self.scores_plots_folder):
            raise Exception("Not a valid path! 2")

        self.all_csv = self.scores_tables_folder + "/all_test.csv"
        self.all_plot = self.scores_plots_folder + "/ScoresUsingAllData.pdf"

        if self.k is not None:
            self.file_tag = str(self.k) + "fold:" + str(self.number_runs) + "runs"
            self.average_plot_title = "Scores Using Repeated K-Fold Validation. K = " + str(self.k) + " N = " + str(
                self.number_runs)

            self.average_plot_pdf = self.scores_plots_folder + "/average" + self.file_tag + ".pdf"
            self.average_csv = self.scores_tables_folder + "/averagemodel scores " + self.file_tag + ".csv"
            self.p_value_csv = self.scores_tables_folder + "/pvalues " + self.file_tag + ".csv"
            self.number_of_wins_csv = self.scores_tables_folder + "/numberofwins " + self.file_tag + ".csv"

            # Df of results from each fold
            self.overall_folds_csv = self.scores_tables_folder + "/overall folds " + self.file_tag + ".csv"
            self.avg_folds_csv = self.scores_tables_folder + "/avg folds " + self.file_tag + ".csv"
        if self.features_to_test is not None:
            self.comparison_csv = self.scores_tables_folder + "/repeatedcomparisons " + self.file_tag + ".csv"
            self.feature_removed_average_csv = dict()
            for feature in self.features_to_test:
                self.feature_removed_average_csv[
                    feature] = self.scores_tables_folder + "/averagemodel scores " + self.file_tag + " " + feature + "removed.csv"

    def prepare_comparison_dicts(self):
        """Summary
        """
        # Dealing with these values could be improved..
        # Counts to compare features
        self.count_without_feature_better = dict()
        self.count_with_feature_better = dict()

        if self.features_to_test is not None:
            for feature in self.features_to_test:
                self.count_without_feature_better[feature] = dict()
                self.count_with_feature_better[feature] = dict()
                for p in self.test_prepositions + ["Average", "Overall"]:
                    self.count_without_feature_better[feature][p] = 0
                    self.count_with_feature_better[feature][p] = 0

    def prepare_folds_dict(self):
        """Generate dictionary to store scores for each fold"""
        folds_dict = dict()
        for model_name in self.model_name_list:
            folds_dict[model_name] = []
        our_model_feature_folds_dict = dict()
        our_model_without_feature_folds_dict = dict()
        if self.features_to_test is not None:
            for feature in self.features_to_test:
                our_model_feature_folds_dict[feature] = dict()
                our_model_without_feature_folds_dict[feature] = dict()
                for p in self.test_prepositions + ["Average", "Overall"]:
                    our_model_feature_folds_dict[feature][p] = []
                    our_model_without_feature_folds_dict[feature][p] = []
        return folds_dict, our_model_feature_folds_dict, our_model_without_feature_folds_dict

    def generate_models(self, train_scenes, test_scenes, extra_features_to_remove=None):
        """Summary
        Generates a list of models to test from given train and test scenes.

        """

        if self.features_to_test is not None:
            # Only test our model
            generate_models = self.model_generator(train_scenes, test_scenes, self.study_info,
                                                   test_prepositions=self.test_prepositions,
                                                   extra_features_to_remove=extra_features_to_remove,
                                                   only_test_our_model=True)

        else:
            # Test all models
            generate_models = self.model_generator(train_scenes, test_scenes, self.study_info,
                                                   test_prepositions=self.test_prepositions)

        return generate_models

    def test_all_scenes(self):
        """Summary
        """
        generate_models = self.Generate_Models_all_scenes
        models = generate_models.models
        t = TestModels(models, "all")
        self.all_dataframe = t.score_dataframe

        self.all_dataframe.to_csv(self.all_csv)

        self.plot_dataframe_bar_chart(self.all_dataframe, self.all_plot, "Preposition", "Score",
                                      "Scores Using All Data")

    def single_validation_test(self, train_scenes, test_scenes):
        """Summary

        Args:
            train_scenes (TYPE): Description
            test_scenes (TYPE): Description
        """
        generate_models = self.generate_models(train_scenes, test_scenes)

        t = TestModels(generate_models.models, str(self.run_count))
        # Get generated scores
        dataset = t.score_dataframe

        # Add scores to total
        if "all_features" in self.dataframe_dict:
            self.dataframe_dict["all_features"] = self.dataframe_dict["all_features"].add(dataset)

        else:
            self.dataframe_dict["all_features"] = dataset

        # Compare Models
        if self.compare is not None:
            # Get our score from dataframe
            # our_score = dataset.at["Overall", self.model_to_test_name]
            for model in self.model_name_list:
                # Get score
                overall_score = dataset.at["Overall", model]

                self.overall_folds_dict[model].append(overall_score)

                avg_score = dataset.at["Average", model]

                self.avg_folds_dict[model].append(avg_score)

        # Add scores to dataframe
        if self.features_to_test is not None:

            for feature in self.features_to_test:
                generate_models_without_feature = self.generate_models(train_scenes, test_scenes,
                                                                       extra_features_to_remove=[feature])

                t = TestModels(generate_models_without_feature.models, str(self.run_count))

                feature_dataset = t.score_dataframe
                # feature_dataset = feature_dataset.drop(["Total Constraint Weights"],axis=1)

                for p in self.test_prepositions + ["Average", "Overall"]:
                    without_feature_score = feature_dataset.at[p, self.model_to_test_name]
                    with_feature_score = dataset.at[p, self.model_to_test_name]

                    if without_feature_score > with_feature_score:
                        self.count_without_feature_better[feature][p] += 1
                    if with_feature_score > without_feature_score:
                        self.count_with_feature_better[feature][p] += 1

                    self.our_model_feature_folds_dict[feature][p].append(with_feature_score)
                    self.our_model_without_feature_folds_dict[feature][p].append(without_feature_score)

                # Add to totals
                if feature in self.dataframe_dict:
                    self.dataframe_dict[feature] = self.dataframe_dict[feature].add(feature_dataset)

                else:
                    self.dataframe_dict[feature] = feature_dataset

    def get_validation_scene_split(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Get train-test scenes

        if self.k is not None:
            # Create random folds for testing
            folds = []

            scenes_left = self.scene_list
            divisor = self.k
            while divisor > 1:
                t_size = float(1) / divisor
                train_scenes, test_scenes = train_test_split(scenes_left, test_size=t_size)
                folds.append(test_scenes)
                scenes_left = train_scenes
                divisor = divisor - 1
                if divisor == 1:
                    folds.append(train_scenes)

            if len(folds) == 0:
                raise Exception("No folds generated")

            return folds

    def folds_check(self, folds):
        """Summary

        Args:
            folds (TYPE): Description

        Returns:
            TYPE: Description
        """

        for f in folds:

            # Check all folds have some constraints to test
            for preposition in self.test_prepositions:

                all_constraints = self.constraint_dict[preposition]

                constraints = []

                for c in all_constraints:
                    if c.scene in f:
                        constraints.append(c)
                if len(constraints) == 0:
                    return False

        for model in self.Generate_Models_all_scenes.models:
            model.folds_check(folds)

        return True

    def validation(self):
        """Summary
        """
        # Perform Repeated random sub-sampling validation
        # Either using k-fold or standard method
        for i in range(self.number_runs):
            self.run_count = i

            print(("Run Number:" + str(i + 1)))
            run_start_time = time.time()

            if self.k is not None:
                # This handles the case where test_scenes do not produce any constraints
                while True:
                    folds = self.get_validation_scene_split()

                    if self.folds_check(folds):
                        for f in folds:
                            print(("Fold Number:" + str(folds.index(f))))
                            fold_start_time = time.time()
                            test_scenes = f
                            train_scenes = []
                            for train_fold in folds:
                                if train_fold != f:
                                    for t in train_fold:
                                        train_scenes.append(t)
                            self.single_validation_test(train_scenes, test_scenes)
                            print("Fold took {} seconds".format(time.time() - fold_start_time))
                        break
                    else:
                        pass
                    # print("Fold with no constraints to test. Retrying...")
            print("Run took {} seconds".format(time.time() - run_start_time))

        # First update value of number of runs to account for folds
        if self.k is not None:
            self.total_number_runs = self.number_runs * self.k
        else:
            self.total_number_runs = self.number_runs
        # Output comparison of models and p-value
        if self.compare is not None:
            # Output folds
            folds_df = pd.DataFrame(self.overall_folds_dict)
            folds_df.to_csv(self.overall_folds_csv)

            folds_df = pd.DataFrame(self.avg_folds_dict)
            folds_df.to_csv(self.avg_folds_csv)

            # Calculate all p values
            # Read --- model1 in column x model2 in row is pvalue model1 is better than model2
            p_value_dfs = []
            for model1 in self.model_name_list:
                model1_dict = dict()
                for model2 in self.model_name_list:
                    if model1 == model2:
                        p_value = 0
                    else:
                        model1_folds = self.overall_folds_dict[model1]
                        model2_folds = self.overall_folds_dict[model2]
                        # Use of wilcoxon is recommended for this sort of test in
                        #  https://towardsdatascience.com/validating-your-machine-learning-model-25b4c8643fb7
                        # Though it may be unsafe as the samples are not independent.
                        T, p_value = wilcoxon(model1_folds, model2_folds, alternative='greater')
                    model1_dict[model2] = p_value
                model1_p_value_df = pd.DataFrame(model1_dict, [model1])
                p_value_dfs.append(model1_p_value_df)
            self.p_value_df = p_value_dfs[0].append(p_value_dfs[1:], sort=False)

            # get number of wins
            # Read --- model1 in column x model2 in row is number of times model1 is better than model2
            win_dfs = []
            for model1 in self.model_name_list:
                model1_dict = dict()
                for model2 in self.model_name_list:
                    model1_folds = self.overall_folds_dict[model1]
                    model2_folds = self.overall_folds_dict[model2]
                    subtracted_folds = list(map(operator.sub, model1_folds, model2_folds))
                    model1_wins = 0
                    for f in subtracted_folds:
                        if f > 0:
                            model1_wins += 1

                    model1_dict[model2] = model1_wins
                model1_win_df = pd.DataFrame(model1_dict, [model1])
                win_dfs.append(model1_win_df)
            self.number_of_wins_df = win_dfs[0].append(win_dfs[1:], sort=False)

        if self.features_to_test is not None:
            feature_p_value = dict()
            with_feature_better = dict()
            without_feature_better = dict()
            for feature in self.features_to_test:
                for p in self.test_prepositions + ["Average", "Overall"]:
                    with_feature_folds = self.our_model_feature_folds_dict[feature][p]
                    without_feature_folds = self.our_model_without_feature_folds_dict[feature][p]

                    if with_feature_folds != without_feature_folds:

                        T, p_value = wilcoxon(with_feature_folds, without_feature_folds, alternative='greater')
                    else:
                        p_value = 0
                    feature_p_value[feature + ":" + p] = p_value
                    with_feature_better[feature + ":" + p] = self.count_with_feature_better[feature][p]
                    without_feature_better[feature + ":" + p] = self.count_without_feature_better[feature][p]

            # Create dataframes to output
            p_value_df = pd.DataFrame(feature_p_value, ["p_value"])
            win_count = pd.DataFrame(with_feature_better, ["With feature wins"])
            lose_count = pd.DataFrame(without_feature_better, ["Without feature wins"])
            # Append dataframes into one
            new_df = p_value_df.append([win_count, lose_count], sort=False)
            self.feature_comparison_df = new_df

        # Print some info
        print(("Total Runs:" + str(self.total_number_runs)))

        # Finalise by averaging scores in dataframe
        for key in self.dataframe_dict:
            self.dataframe_dict[key] = self.dataframe_dict[key].div(self.total_number_runs)

    def output(self):
        """Summary
        """
        # # Edit plot settings
        mpl.rcParams['font.size'] = 40
        mpl.rcParams['legend.fontsize'] = 37
        mpl.rcParams['axes.titlesize'] = 'medium'
        mpl.rcParams['axes.labelsize'] = 'medium'
        mpl.rcParams['ytick.labelsize'] = 'small'
        # Handle outputting here so we're not always outputting
        self.average_dataframe = self.dataframe_dict["all_features"]
        # Reorder columns for output
        if self.features_to_test == None:
            new_column_order = self.model_name_list
            reordered_df = self.average_dataframe[new_column_order]
            reordered_df.to_csv(self.average_csv)
        else:
            self.average_dataframe.to_csv(self.average_csv)

        self.plot_dataframe_bar_chart(self.average_dataframe, self.average_plot_pdf, "Preposition", "Score",
                                      self.average_plot_title)
        if self.compare is not None:
            # Output to csv
            self.p_value_df.to_csv(self.p_value_csv)
            self.number_of_wins_df.to_csv(self.number_of_wins_csv)

        if self.features_to_test is not None:

            # Output to csv
            self.feature_comparison_df.to_csv(self.comparison_csv)
            for feature in self.features_to_test:
                dff = self.dataframe_dict[feature]

                dff.to_csv(self.feature_removed_average_csv[feature])

            out = dict()

            for feature in self.features_to_test:
                print((self.dataframe_dict[feature]))
                out[feature] = self.dataframe_dict[feature][self.model_to_test_name]
            out["None removed"] = self.average_dataframe[self.model_to_test_name]
            df = pd.DataFrame(out, self.test_prepositions + ["Average", "Overall"])
            self.functional_feature_analysis_df = df
            df.to_csv(self.scores_tables_folder + "/functional_feature_analysis.csv")

            output_file = self.scores_plots_folder + "/ScoresWithRemovedFeatures.pdf"
            x_label = "Preposition"
            y_label = "Score"
            plot_title = "Average Scores With Removed Features. K = " + str(self.k) + " N = " + str(self.number_runs)
            self.plot_dataframe_bar_chart(df, output_file, x_label, y_label, plot_title)

    def plot_dataframe_bar_chart(self, dataset, file_to_save, x_label, y_label, plot_title):
        """Summary

        Args:
            dataset (TYPE): Description
            file_to_save (TYPE): Description
            x_label (TYPE): Description
            y_label (TYPE): Description
            plot_title (TYPE): Description
        """
        if self.features_to_test == None:
            new_column_order = self.model_name_list
            reordered_df = dataset[new_column_order]
        else:
            reordered_df = dataset

        ax = reordered_df.plot(kind='bar', width=0.85, title=plot_title, figsize=(20, 10), legend=True)

        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label)
        ax.set_yticks(np.arange(0, 1.01, 0.05))
        ax.set_ylim([0, 1])
        ax.set_title(plot_title, pad=10)
        ax.grid(True)
        ax.set_axisbelow(True)

        plt.legend(loc='upper center', bbox_to_anchor=(0.44, -0.42), ncol=3)

        # plt.show()
        plt.savefig(file_to_save, bbox_inches='tight')

    def plot_bar_from_csv(self, file, file_to_save, x_label, y_label, plot_title, columns_to_drop=None):
        """Summary

        Args:
            file (TYPE): Description
            file_to_save (TYPE): Description
            x_label (TYPE): Description
            y_label (TYPE): Description
            plot_title (TYPE): Description
            columns_to_drop (None, optional): Description
        """
        dataset = pd.read_csv(file, index_col=0)
        if columns_to_drop is not None:
            dataset = dataset.drop(columns_to_drop, axis=1)
        self.plot_dataframe_bar_chart(dataset, file_to_save, x_label, y_label, plot_title)


def compare_models(runs, k, model_generator, base_output_folder, test_prepositions=PREPOSITION_LIST):
    study_info = StudyInfo("2019 study")

    m = MultipleRuns(model_generator, base_output_folder + "tables",
                     base_output_folder + "plots", study_info, test_prepositions=test_prepositions,
                     number_runs=runs,
                     k=k,
                     compare="y")
    models_to_test = m.Generate_Models_all_scenes.models

    t = TestModels(models_to_test, "all")
    all_dataframe = t.score_dataframe.copy()
    print(all_dataframe)
    all_dataframe.to_csv(base_output_folder + "/tables/initial_test.csv")

    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print(m.average_dataframe)


class ModelGenerator:
    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions):
        """Summary

        Generic model generator. Init of model generators should return model self.models and self.model_names_list
        """
        self.study_info = study_info_

        # Scenes used to train models
        self.train_scenes = train_scenes
        # Scenes used to test models
        self.test_scenes = test_scenes

        # Make sure train and test scenes are distinct, if not using all scenes
        if self.train_scenes != self.study_info.scene_name_list:
            f1_set = set(self.train_scenes)
            f2_set = set(self.test_scenes)
            if f1_set & f2_set:
                raise ValueError('Train and test scenes not distinct')

        # Features to remove from consideration (not used in training or testing)
        self.features_to_remove = Configuration.object_specific_features.copy()
        self.test_prepositions = test_prepositions

    def generate_model_lists(self):

        self.models = [self.__getattribute__(x) for x in self.__dir__() if isinstance(self.__getattribute__(x), Model)]

        self.model_name_list = [m.name for m in self.models]


class TestModels:
    """Summary
    # Takes input set of models, gets their scores and creates a dataframe of scores.
    Attributes:
        model_name_list (list): Description
        models (TYPE): Description
        score_dataframe (TYPE): Description
        version_name (TYPE): Description
    """

    def __init__(self, models, version_name):
        """Summary

        Args:
            models (TYPE): Description
            version_name (TYPE): Description
        """
        self.version_name = version_name
        self.models = models
        self.model_name_list = []

        out = dict()

        for model in self.models:
            self.model_name_list.append(model.name)

            model.get_score()
            out[model.name] = model.scores

            print(model.name)
            print(model.scores)

        # out["Total Constraint Weights"] = models[0].weight_totals + ["",""]

        df = pd.DataFrame(out, self.models[0].test_prepositions + ["Average", "Overall"])

        # Reorder columns
        new_column_order = self.model_name_list
        reordered_df = df[new_column_order]

        self.score_dataframe = reordered_df

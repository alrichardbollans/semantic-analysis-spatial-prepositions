"""Summary
This file provides classes for generating models of typicality and running tests on them.
First run compile_instances.py

Attributes:
    COMP_FILETAG (TYPE): Description
    PREPOSITION_LIST (TYPE): Description
    SV_FILETAG (TYPE): Description
"""

# Standard imports

import pandas as pd
import numpy as np
import math

# Modules for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Modules for testing and model making
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# Local module imports
from typing import Dict

from Analysis.performance_test_functions import ModelGenerator, MultipleRuns, Model, BASIC_MODEL_SCORES_FOLDER, \
    BASIC_MODEL_PROPERTY_FOLDER
from compile_instances import InstanceCollection, SemanticCollection, ComparativeCollection
from data_import import Configuration, StudyInfo

# Useful global variables
SV_FILETAG = SemanticCollection.filetag  # Tag for sv task files
COMP_FILETAG = ComparativeCollection.filetag  # Tag for comp task files
PREPOSITION_LIST = StudyInfo.preposition_list


def rename_feature(feature):
    # Rename some features
    new_name = feature
    if new_name == "contact_proportion":
        new_name = "contact"
    if new_name == "bbox_overlap_proportion":
        new_name = "containment"

    return new_name


def convert_index(i, number_of_columns):
    """Converts index to place in row/columns for plotting subplots
    
    Args:
        i (int): Description
        number_of_columns (int): Description
    
    Returns:
        TYPE: Description
    """
    if i == 0 or i == 6 or i == 12:
        x_pos = 0
        y_pos = 0
    elif i < 6:
        x_pos = int(i / number_of_columns)
        y_pos = i % number_of_columns
    else:
        i = i - 6
        x_pos = int(i / number_of_columns)
        y_pos = i % number_of_columns

    return x_pos, y_pos


class SemanticMethods:
    """Class for reusable methods related to semantic measurements
    """

    @staticmethod
    def semantic_distance(weight_array, x, y):
        """
        Parameters:
            weight_array: 1-D Array of feature weights
            x (TYPE): Description
            y: 1-D arrays. Points to compare
            feature_keys (TYPE): Description
            features_to_remove: Feature to remove from consideration
        
        Returns:
            distance: Float representing semantic distance from list_of_annotations to y

        
        Deleted Parameters:
            list_of_annotations: 1-D arrays. Points to compare
        """

        point = np.subtract(x, y)  # Subtract arrays point wise

        point = np.square(point)  # Square pointwise

        summ = np.dot(point, weight_array)  # Dot product pointwise by weights

        distance = math.sqrt(summ)  # Square root to get distance

        return distance

    @staticmethod
    def semantic_similarity(weight_array, x, y):
        """
        Calculates weighted semantic similarity between x and y.

        :param weight_array:
        :param x:
        :param y:
        :return: float
        """

        # Get semantic distance
        distance = SemanticMethods.semantic_distance(weight_array, x, y)
        # Get similarity
        out = math.exp(-distance)
        return out


class GeneratePrepositionModelParameters:
    """Summary

    Class with methods to generate model parameters for particular preposition
    
    Attributes:

    """

    # Given training scenes, works out models for individual preposition
    # Must call work_out_models() to generate the parameters
    ratio_feature_name = InstanceCollection.ratio_feature_name
    categorisation_feature_name = InstanceCollection.categorisation_feature_name
    scene_feature_name = InstanceCollection.scene_feature_name
    fig_feature_name = InstanceCollection.fig_feature_name
    ground_feature_name = InstanceCollection.ground_feature_name

    ratio_index = -2
    category_index = -1
    scene_index = 0

    interval = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(-1, 1)

    # degree to use in polynomial regression
    polynomial_degree = 3

    def __init__(self, study_info_, preposition, train_scenes, features_to_remove=None, polyseme=None,
                 given_dataset=None, oversample: bool = False):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
            train_scenes (TYPE): Description
            features_to_remove (None, optional): Description
            polyseme (None, optional): Description
        """

        if features_to_remove is None:
            print("No features to remove given. Are you sure?")
            features_to_remove = []

        self.study_info = study_info_

        self.all_feature_keys = self.study_info.all_feature_keys
        self.features_to_remove = features_to_remove
        self.oversample = oversample

        self.feature_keys = []
        for f in self.all_feature_keys:
            if f not in self.features_to_remove:
                self.feature_keys.append(f)

        self.train_scenes = train_scenes

        # Given polyseme if being used to model polyseme
        self.polyseme = polyseme

        self.preposition = preposition
        # Use pandas dataframes for ease of importing etc..
        # Created in compile_instances write_config_ratios()
        # Row is created in dataset only if the configuration was tested
        # Set of values with selection information
        # Only includes configurations that were tested at least once
        config_ratio_csv = self.study_info.config_ratio_csv(SV_FILETAG, preposition)
        if given_dataset is None:
            self.dataset = pd.read_csv(config_ratio_csv)
        else:
            self.dataset = given_dataset

        if self.oversample:
            # Oversampling
            # The data is first oversampled to improve categorisation of (rare) positive instances.
            copy_df = self.dataset.copy()
            positive_examples = copy_df[(copy_df.iloc[:, self.category_index] == 1)]
            oversampled_df = pd.concat([copy_df, positive_examples], ignore_index=True)
            oversampled_df = pd.concat([oversampled_df, positive_examples], ignore_index=True)
            self.dataset = oversampled_df

        if self.polyseme is not None:

            # # Remove none polyseme preposition instances from dataset
            indexes_to_drop = []
            indexes_to_drop_pid = []
            for index, row in self.dataset.iterrows():
                # remove non feature values from row
                value_row = row.values[3:-2]

                potential_instance = polyseme.potential_instance(value_row)

                if not (potential_instance):
                    indexes_to_drop_pid.append(index)
                if not (potential_instance) and row[self.categorisation_feature_name] != 0:
                    indexes_to_drop.append(index)

            self.possible_instances_dataset = self.dataset.copy()
            # Dataset to train polyseme on (configurations not yet removed for training)
            self.dataset.drop(self.dataset.index[indexes_to_drop], inplace=True)
            # Dataset of configurations that fit polyseme conditions

            self.possible_instances_dataset.drop(self.possible_instances_dataset.index[indexes_to_drop_pid],
                                                 inplace=True)
            # Remove non-training instances
            self.train_possible_instances_dataset = self.remove_nontrainingscenes(
                self.possible_instances_dataset)

        # Remove rows from dataset where not training scene
        self.train_dataset = self.remove_nontrainingscenes(
            self.dataset)

        # Remove selection info columns and names to only have features
        self.allFeatures = self.remove_nonfeatures(self.train_dataset)

        # Feature dataframe without unused features for training the models.
        self.feature_dataframe = self.remove_unused_features(self.allFeatures)

        # Remove rows from above where not a preposition instance
        self.aff_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.category_index] == 1)]

        # Remove selection info columns to only have features
        self.affAllFeatures = self.remove_nonfeatures(self.aff_dataset)
        # Feature dataframe of all possible instances
        self.affFeatures = self.remove_unused_features(self.affAllFeatures)

        # Remove rows from training set where not a 'good' preposition instance
        self.good_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.ratio_index] >= 0.5)]

        # Remove selection info columns to only have features
        self.goodAllFeatures = self.remove_nonfeatures(self.good_dataset)
        # Feature dataframe of all possible instances
        self.goodFeatures = self.remove_unused_features(self.goodAllFeatures)

        # Typical instances are the best examples
        ratio_max = self.train_dataset[self.ratio_feature_name].max()
        self.typical_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.ratio_index] == ratio_max)]
        # Remove selection info columns to only have features
        self.typical_features = self.remove_nonfeatures(self.typical_dataset)
        # Ratio dataset with non-instances - nobody selected the preposition
        self.neg_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.category_index] == 0)]
        # Remove selection info columns to only have features
        self.neg_features = self.remove_nonfeatures(self.neg_dataset)

        # prototype calculated using regression. Stored as array
        self.prototype = []

        self.prototype_csv = BASIC_MODEL_PROPERTY_FOLDER + "prototypes/" + preposition + ".csv"

        # regression weights calculated by linear regression. stored as array and dataframe
        self.poly_regression_model = None

        self.linear_regression_model = None

        self.ridge_regression_model = None

        self.regression_weights = []
        self.regression_weight_csv = BASIC_MODEL_PROPERTY_FOLDER + "regression weights/" + preposition + ".csv"
        self.all_features_regression_weight_csv = BASIC_MODEL_PROPERTY_FOLDER + "regression weights/allfeatures_" + preposition + ".csv"

        # Stores model predictions for later plotting
        self.interval_predictions = dict()

        # barycentre_prototype . stored as array
        self.barycentre_prototype = None

        self.barycentre_csv = BASIC_MODEL_PROPERTY_FOLDER + "barycentre model/" + preposition + "-prototype.csv"

        # exemplar_mean . stored as array
        self.exemplar_mean = None

        self.exemplar_csv = BASIC_MODEL_PROPERTY_FOLDER + "exemplar/" + preposition + "-exemplar_means.csv"

    def remove_nontrainingscenes(self, d):
        """Summary
        Removes instances from non training scenes.
        Args:
            d (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        copy_d = d.copy()
        return copy_d[(copy_d.iloc[:, self.scene_index].isin(self.train_scenes))]

    def remove_nonfeatures(self, d):
        """Summary
        Removes nonfeature columns from a dataframe.
        Args:
            d (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Remove selection info columns and names to only have features
        return d.drop(["Scene", "Figure", "Ground", self.ratio_feature_name, self.categorisation_feature_name], axis=1)

    def remove_unused_features(self, d):
        """Summary
        Drops unused feature columns from dataframe.
        Args:
            d (TYPE): Description

        Returns:
            TYPE: Description
        """
        new_d = d.copy()

        if self.features_to_remove is not None:
            # Remove features to remove
            new_d = new_d.drop(self.features_to_remove, axis=1)

            if len(self.features_to_remove) == 0:
                print("No features being removed.")
        return new_d

    def remove_unused_features_from_point_array(self, point):
        new_point = []
        for f in self.all_feature_keys:
            if f not in self.features_to_remove:
                new_point.append(point[self.all_feature_keys.index(f)])

        return new_point

    def work_out_models(self):
        """Summary
        """
        self.work_out_linear_regression_model()
        self.work_out_barycentre_prototype()
        self.work_out_exemplar_mean()
        self.work_out_prototype_model()

    def get_feature_index_in_array(self, feature):
        '''Returns index for feature in prototype/weight arrays etc.
        '''
        return self.all_feature_keys.index(feature)

    def remove_unused_features_from_array(self, array):
        '''
        Takes an array of all feature values and returns array of only used feature values.
        :param array:
        :return:
        '''
        new_array = []
        for f in self.all_feature_keys:
            if f in self.feature_keys:
                new_array.append(array[self.all_feature_keys.index(f)])

        return np.array(new_array)

    def work_out_barycentre_prototype(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        X = self.affAllFeatures

        for feature in self.all_feature_keys:
            pro_value = X[feature].mean()

            out.append(pro_value)

        out = np.array(out)
        self.barycentre_prototype = out
        return self.barycentre_prototype

    def work_out_exemplar_mean(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        X = self.typical_features

        for feature in self.all_feature_keys:
            pro_value = X[feature].mean()
            out.append(pro_value)

        out = np.array(out)
        self.exemplar_mean = out
        return self.exemplar_mean

    def work_out_feature_prototype(self, feature):
        """Summary

        Args:
            feature (TYPE): Description

        Returns:
            TYPE: Description
        """
        # First predict feature value given selection ratio of 1
        # Reshape data first
        # This should be capped but hasn't been
        X = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)
        Y = self.train_dataset[feature].values.reshape(-1, 1)

        model1 = LinearRegression()

        # Fit model to data
        model1.fit(X, Y)
        y_pred = model1.predict(X)
        self.interval_predictions[feature] = y_pred

        # Get prototype for feature
        max_point = np.array([1]).reshape(-1, 1)

        feature_prototype = model1.predict(max_point)

        pro_value = feature_prototype[0][0]
        return pro_value

    def work_out_ridge_regression(self):
        X = self.feature_dataframe
        Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)

        ridge_model = Ridge()

        # Fit model to data
        ridge_model.fit(X, Y)

        self.ridge_regression_model = ridge_model

        return ridge_model

    def work_out_linear_regression_model(self):
        """Summary

        Returns:
            TYPE: Description
        """

        # Reshape data first

        X = self.feature_dataframe
        Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)

        lin_model = LinearRegression()

        # Fit model to data
        lin_model.fit(X, Y)

        self.linear_regression_model = lin_model

        return lin_model

    def work_out_feature_weights(self):
        """Summary
        """
        if self.linear_regression_model is None:
            model2 = self.work_out_linear_regression_model()
        else:
            model2 = self.linear_regression_model

        X = self.feature_dataframe

        v = pd.DataFrame(model2.coef_, index=["coefficient"]).transpose()
        w = pd.DataFrame(X.columns, columns=["feature"])
        coeff_df = pd.concat([w, v], axis=1, join="inner")
        coeff_df = coeff_df.set_index("feature")

        weights_all_features = []
        weights_used_features = []

        for feature in self.all_feature_keys:
            if self.features_to_remove is not None:
                # If the feature is removed, append 0 instead
                if feature in self.features_to_remove:
                    weights_all_features.append(0)
                else:
                    w = abs(coeff_df.loc[feature, "coefficient"])
                    weights_all_features.append(w)
                    weights_used_features.append(w)
            else:
                w = abs(coeff_df.loc[feature, "coefficient"])
                weights_all_features.append(w)
                weights_used_features.append(w)

        # Create two separate arrays. One where unused features are given weight 0 and
        # one where unused features are not added. The latter is only used when finding
        # which cluster centre a configuration si closest to in the KMeans Model

        self.regression_weights = np.array(weights_all_features)
        self.regression_weights_used_features = np.array(weights_used_features)

    def work_out_prototype_model(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Work out linear regression on each feature by comparing to the ratio of times selected

        # This step gives prototypes for later steps, which are saved to prototypes folder

        prototype = []
        for feature in self.all_feature_keys:
            # First predict feature value given selection ratio of 1
            pro_value = self.work_out_feature_prototype(feature)

            # Add to dictionary
            prototype.append(pro_value)

        self.work_out_feature_weights()

        self.prototype = np.array(prototype)

        return self.prototype

    def work_out_polynomial_regression_model(self):
        """Summary

        Args:


        Returns:
            TYPE: Description
        """
        # Next get gradient when feature predicts selection ratio
        # Reshape data first
        X = self.feature_dataframe
        Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)

        polynomial_features = PolynomialFeatures(degree=self.polynomial_degree)
        x_poly = polynomial_features.fit_transform(X)

        model2 = LinearRegression()

        # Fit model to data
        model2.fit(x_poly, Y)

        self.poly_regression_model = model2

        return model2

    def output_models(self):
        """Summary
        """

        # Only called once when training scenes are all scenes, so these are the best model parameters
        wf = pd.DataFrame(self.regression_weights, self.all_feature_keys)

        wf.to_csv(self.regression_weight_csv)

        pf = pd.DataFrame(self.prototype, self.all_feature_keys)

        pf.to_csv(self.prototype_csv)

        epf = pd.DataFrame(self.barycentre_prototype, self.all_feature_keys)

        epf.to_csv(self.barycentre_csv)

        exf = pd.DataFrame(self.exemplar_mean, self.all_feature_keys)

        exf.to_csv(self.exemplar_csv)

    def read_all_feature_weights(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Read regression weights for all features
        wf = pd.read_csv(self.all_features_regression_weight_csv, index_col=0)

        return wf

    def read_regression_weights(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Read regression weights for relations
        wf = pd.read_csv(self.regression_weight_csv, index_col=0)

        return wf

    def plot_features_ratio_to_axis(self, feature, axis):
        # Reshape data first
        Y = self.train_dataset[feature].values.copy()
        Y = Y.reshape(-1, 1)
        X = self.train_dataset[self.ratio_feature_name].values.copy()
        X = X.reshape(-1, 1)
        # Get prediction of all points on interval
        y_pred = self.interval_predictions[feature]

        feature_processer = self.study_info.feature_processor
        # Convert values back to human readable
        for i in range(len(Y)):
            y = Y[i]
            new_y = feature_processer.convert_standardised_value_to_normal(feature, y)
            Y[i] = new_y

        for i in range(len(y_pred)):
            y = y_pred[i]
            new_y = feature_processer.convert_standardised_value_to_normal(feature, y)
            y_pred[i] = new_y

        axis.set_xlabel("Selection Ratio")
        ylabel = rename_feature(feature)

        axis.set_ylabel(ylabel)

        axis.grid(True)

        # Plot data point scatter
        axis.plot(X, Y, 'k.')
        # Plot regression line
        axis.plot(X, y_pred, color='red', linewidth=2)
        # Plot barycentre and exemplar values
        end = [1]
        end = np.array(end).reshape(-1, 1)

        index_for_prototypes = self.all_feature_keys.index(feature)
        if self.barycentre_prototype is not None:
            b = self.barycentre_prototype[index_for_prototypes]
            b = feature_processer.convert_standardised_value_to_normal(feature, b)
            b = np.array([b]).reshape(-1, 1)
            # Plot barycentre value
            axis.plot(end, b, markersize=10, markeredgewidth=2, marker='+')
        if self.exemplar_mean is not None:
            ex = self.exemplar_mean[index_for_prototypes]
            ex = feature_processer.convert_standardised_value_to_normal(feature, ex)
            ex = np.array([ex]).reshape(-1, 1)

            # Plot exemplar mean value
            axis.plot(end, ex, markersize=10, markeredgewidth=2, marker=(5, 2))

        # # COnvert y labels to human readable values
        # feature_processer = Features(self.study_info.name)
        #
        # vals = axis.get_yticks()
        # print(vals)
        # new_vals = [feature_processer.convert_standardised_value_to_normal(feature, x) for x in vals]
        # print(new_vals)
        # axis.set_yticklabels(new_vals)

    def plot_features_ratio(self, no_columns, axes, feature):
        """Summary

        Args:

        """

        index = self.feature_keys.index(feature)

        # Get position to  display, by index
        x_pos, y_pos = convert_index(index, no_columns)

        ax1 = axes[x_pos, y_pos]

        self.plot_features_ratio_to_axis(feature, ax1)

    def get_plot_filename(self, file_no):
        """Summary
        
        Args:
            file_no (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        x = str(file_no)

        if self.polyseme is not None:
            filename = self.polyseme.plot_folder + self.preposition + "-" + self.polyseme.polyseme_name + x + ' .pdf'
        else:

            filename = BASIC_MODEL_PROPERTY_FOLDER + "plots/" + self.preposition + x + ".pdf"
        return filename

    def plot_models(self, base_folder=None):
        """Summary
        """
        if base_folder is None:
            base_folder = ""
        # Plots simple linear regressions used to find prototypes
        no_rows = 3
        no_columns = 2

        fig, axes = plt.subplots(nrows=no_rows, ncols=no_columns, sharex=False, sharey=False)
        fig.tight_layout()
        fig.canvas.set_window_title('Ratio vs. Feature')
        plot_count = 0
        file_no = 1

        for feature in self.feature_keys:

            plot_count += 1

            r = plot_count % (no_columns * no_rows)

            self.plot_features_ratio(no_columns, axes, feature)

            filename = base_folder + self.get_plot_filename(file_no)

            # When the figure is full of plots, save figure
            if r == 0:
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)
                file_no += 1

                # Clear plots for new figure
                fig, axes = plt.subplots(nrows=no_rows, ncols=no_columns, sharex=False, sharey=False)
                fig.tight_layout()
                fig.canvas.set_window_title('Ratio vs. Feature')
        # Save remaining plots
        filename = base_folder + self.get_plot_filename(file_no)
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def plot_single_feature_regression(self, feature):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.tight_layout()
        fig.canvas.set_window_title('Ratio vs. Feature')
        self.plot_features_ratio_to_axis(feature, axes)
        filename = BASIC_MODEL_PROPERTY_FOLDER + "plots/individual features/" + self.preposition + feature + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def plot_feature_space(self, feature1, feature2):
        feature_processer = self.study_info.feature_processor

        X = self.train_dataset[feature1].values.copy()
        Y = self.train_dataset[feature2].values.copy()

        # Convert values back to human readable
        for i in range(len(X)):
            x = X[i]
            new_x = feature_processer.convert_standardised_value_to_normal(feature1, x)
            X[i] = new_x
        for i in range(len(Y)):
            y = Y[i]
            new_y = feature_processer.convert_standardised_value_to_normal(feature2, y)
            Y[i] = new_y

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        sr = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)

        instances = plt.scatter(X, Y, c=sr)
        plt.xlabel(rename_feature(feature1))
        plt.ylabel(rename_feature(feature2))
        cbar = plt.colorbar()
        cbar.set_label('Selection ratio', rotation=270)
        cbar.set_ticks([0, 1])

        # Get prototype, barycentre and exemplar values for each feature
        index_for_prototypes1 = self.all_feature_keys.index(feature1)
        b1 = feature_processer.convert_standardised_value_to_normal(feature1,
                                                                    self.barycentre_prototype[index_for_prototypes1])
        p1 = feature_processer.convert_standardised_value_to_normal(feature1, self.prototype[index_for_prototypes1])
        ex1 = feature_processer.convert_standardised_value_to_normal(feature1,
                                                                     self.exemplar_mean[index_for_prototypes1])

        index_for_prototypes2 = self.all_feature_keys.index(feature2)
        b2 = feature_processer.convert_standardised_value_to_normal(feature2,
                                                                    self.barycentre_prototype[index_for_prototypes2])
        p2 = feature_processer.convert_standardised_value_to_normal(feature2, self.prototype[index_for_prototypes2])
        ex2 = feature_processer.convert_standardised_value_to_normal(feature2,
                                                                     self.exemplar_mean[index_for_prototypes2])
        marker_size = mpl.rcParams['lines.markersize'] ** 2.5
        b1 = np.array([b1]).reshape(-1, 1)
        b2 = np.array([b2]).reshape(-1, 1)
        # Plot barycentre value
        barycentre = plt.scatter(b1, b2, marker='+', c='red', s=marker_size)

        p1 = np.array([p1]).reshape(-1, 1)
        p2 = np.array([p2]).reshape(-1, 1)
        # Plot prototype value
        prototype = plt.scatter(p1, p2, marker='X', c='red', s=marker_size)

        ex1 = np.array([ex1]).reshape(-1, 1)
        ex2 = np.array([ex2]).reshape(-1, 1)
        # Plot exemplar value
        exemplar = plt.scatter(ex1, ex2, marker='*', c='red', s=marker_size)

        plt.legend((instances, barycentre, exemplar, prototype),
                   ('Instances', 'CS Prototype', 'Exemplar Mean', 'Generated Prototype'),
                   scatterpoints=1,
                   loc='upper center', bbox_to_anchor=(0.5, -0.13),

                   ncol=2,
                   fontsize=15)

        # plt.title("Instances of '" + self.preposition + "'")
        filename = BASIC_MODEL_PROPERTY_FOLDER + "plots/feature spaces/" + self.preposition + feature1 + feature2 + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()


class PrototypeModel(Model):
    name = "Our Prototype"

    def __init__(self, preposition_model_dict: Dict[str, GeneratePrepositionModelParameters], test_scenes,
                 study_info_: StudyInfo, test_prepositions=PREPOSITION_LIST,
                 constraint_csv_removed_users=None):
        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, PrototypeModel.name, test_scenes, study_info_, test_prepositions=test_prepositions,
                       constraint_csv_removed_users=constraint_csv_removed_users)

        if len(test_scenes) < len(study_info_.scene_name_list):
            for p in self.test_prepositions:
                train_scenes = set(self.preposition_model_dict[p].train_scenes)
                if (any(x in train_scenes for x in test_scenes)):
                    raise ValueError("Train and test scene overlap.")

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        p_model = self.preposition_model_dict[preposition]
        weight_array = p_model.regression_weights
        prototype_array = p_model.prototype

        out = SemanticMethods.semantic_similarity(weight_array, value_array, prototype_array)

        return out


class CSModel(Model):
    name = "Conceptual Space"

    def __init__(self, preposition_model_dict, test_scenes, study_info_,
                 constraint_csv_removed_users=None):
        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, CSModel.name, test_scenes, study_info_,
                       constraint_csv_removed_users=constraint_csv_removed_users)

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        p_model = self.preposition_model_dict[preposition]
        weight_array = p_model.regression_weights
        prototype_array = p_model.barycentre_prototype
        out = SemanticMethods.semantic_similarity(weight_array, value_array, prototype_array)

        return out


class ExemplarModel(Model):
    name = "Exemplar"

    def __init__(self, preposition_model_dict, test_scenes, study_info_,
                 constraint_csv_removed_users=None):

        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, ExemplarModel.name, test_scenes, study_info_,
                       constraint_csv_removed_users=constraint_csv_removed_users)

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):

        p_model = self.preposition_model_dict[preposition]
        weight_array = p_model.regression_weights

        exemplars = p_model.typical_features
        none_instances = p_model.neg_features

        # Find average semantic similarity to points in exemplar model
        counter = 0
        semantic_similarity_sum = 0

        # Iterate over rows in exemplar dataframe
        for index, row in exemplars.iterrows():
            # Get row values
            exemplar_values = row.values

            # Convert values to np array
            exemplar_values = np.array(exemplar_values)
            counter += 1
            # Calculate similarity of current point to exemplar

            semantic_similarity_sum += SemanticMethods.semantic_similarity(weight_array, value_array, exemplar_values)

        if counter == 0:
            return 0
        else:
            average = float(semantic_similarity_sum) / counter

        return average


class ProximityModel(Model):
    name = "Proximity"

    def __init__(self, test_scenes, study_info_,
                 constraint_csv_removed_users=None):

        Model.__init__(self, ProximityModel.name, test_scenes, study_info_,
                       constraint_csv_removed_users=constraint_csv_removed_users)

        prototype_array = []
        weight_array = []

        for feature in self.all_feature_keys:
            if feature != "shortest_distance":
                prototype_array.append(0)
                weight_array.append(0)
            else:
                x = self.feature_processer.convert_normal_value_to_standardised(feature, 0)
                prototype_array.append(x)
                weight_array.append(1)

        self.prototype_array = np.array(prototype_array)
        self.weight_array = np.array(weight_array)

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):

        out = SemanticMethods.semantic_similarity(self.weight_array, value_array, self.prototype_array)

        return out


class SimpleModel(Model):
    name = "Simple"

    def __init__(self, test_scenes, study_info_,
                 constraint_csv_removed_users=None):
        Model.__init__(self, SimpleModel.name, test_scenes, study_info_,
                       constraint_csv_removed_users=constraint_csv_removed_users)

        self.prototype_dictionary = dict()
        self.weight_dictionary = dict()

        in_prototype, in_weights = self.generate_arrays({"bbox_overlap_proportion": 1})
        self.prototype_dictionary["in"] = self.prototype_dictionary["inside"] = in_prototype
        self.weight_dictionary["in"] = self.weight_dictionary["inside"] = in_weights

        on_prototype, on_weights = self.generate_arrays({"above_proportion": 1, "contact_proportion": 1})
        self.prototype_dictionary["on"] = self.prototype_dictionary["on top of"] = on_prototype
        self.weight_dictionary["on"] = self.weight_dictionary["on top of"] = on_weights

        above_prototype, above_weights = self.generate_arrays({"above_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["above"] = self.prototype_dictionary["over"] = above_prototype
        self.weight_dictionary["above"] = self.weight_dictionary["over"] = above_weights

        below_prototype, below_weights = self.generate_arrays({"below_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["below"] = self.prototype_dictionary["under"] = below_prototype
        self.weight_dictionary["below"] = self.weight_dictionary["under"] = below_weights

        against_prototype, against_weights = self.generate_arrays({"contact_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["against"] = against_prototype
        self.weight_dictionary["against"] = against_weights

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        prototype_array = self.prototype_dictionary[preposition]
        weight_array = self.weight_dictionary[preposition]

        out = SemanticMethods.semantic_similarity(weight_array, value_array, prototype_array)

        return out


class BestGuessModel(Model):
    name = "Best Guess"

    def __init__(self, test_scenes, study_info_,
                 constraint_csv_removed_users=None):
        Model.__init__(self, BestGuessModel.name, test_scenes, study_info_,
                       constraint_csv_removed_users=constraint_csv_removed_users)

        self.prototype_dictionary = dict()
        self.weight_dictionary = dict()

        inside_prototype, inside_weights = self.generate_arrays({"bbox_overlap_proportion": 1})
        self.prototype_dictionary["inside"] = inside_prototype
        self.weight_dictionary["inside"] = inside_weights

        in_prototype, in_weights = self.generate_arrays({"bbox_overlap_proportion": 1, "location_control": 1})
        self.prototype_dictionary["in"] = in_prototype
        self.weight_dictionary["in"] = in_weights

        on_prototype, on_weights = self.generate_arrays({"above_proportion": 1, "contact_proportion": 1, "support": 1})
        self.prototype_dictionary["on"] = on_prototype
        self.weight_dictionary["on"] = on_weights

        ontopof_prototype, ontopof_weights = self.generate_arrays({"above_proportion": 1, "contact_proportion": 1})
        self.prototype_dictionary["on top of"] = ontopof_prototype
        self.weight_dictionary["on top of"] = ontopof_weights

        above_prototype, above_weights = self.generate_arrays({"above_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["above"] = above_prototype
        self.weight_dictionary["above"] = above_weights

        over_prototype, over_weights = self.generate_arrays({"above_proportion": 1, "f_covers_g": 1})
        self.prototype_dictionary["over"] = over_prototype
        self.weight_dictionary["over"] = over_weights

        below_prototype, below_weights = self.generate_arrays({"below_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["below"] = below_prototype
        self.weight_dictionary["below"] = below_weights

        under_prototype, under_weights = self.generate_arrays({"below_proportion": 1, "g_covers_f": 1})
        self.prototype_dictionary["under"] = under_prototype
        self.weight_dictionary["under"] = under_weights

        against_prototype, against_weights = self.generate_arrays(
            {"contact_proportion": 1, "horizontal_distance": 0, "location_control": 0.5})
        self.prototype_dictionary["against"] = against_prototype
        self.weight_dictionary["against"] = against_weights

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):
        prototype_array = self.prototype_dictionary[preposition]
        weight_array = self.weight_dictionary[preposition]

        out = SemanticMethods.semantic_similarity(weight_array, value_array, prototype_array)

        return out


class GenerateBasicModels(ModelGenerator):
    """Summary
    

    """

    # Generating models to test
    def __init__(self, train_scenes, test_scenes, study_info_, extra_features_to_remove=None, only_test_our_model=None,
                 test_prepositions=PREPOSITION_LIST):
        """Summary
        

        """

        ModelGenerator.__init__(self, train_scenes, test_scenes, study_info_, test_prepositions)

        # Extra features may be removed in order to compare performance
        if extra_features_to_remove is not None:
            for f in extra_features_to_remove:
                self.features_to_remove.append(f)

        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in self.test_prepositions:
            M = GeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
                                                   features_to_remove=self.features_to_remove)
            M.work_out_models()
            preposition_models_dict[p] = M

        self.preposition_parameters_dict = preposition_models_dict

        self.our_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info)

        if only_test_our_model is None:
            self.exemplar_model = ExemplarModel(preposition_models_dict, self.test_scenes, self.study_info)
            self.cs_model = CSModel(preposition_models_dict, self.test_scenes, self.study_info)
            self.proximity_model = ProximityModel(self.test_scenes, self.study_info)
            self.simple_model = SimpleModel(self.test_scenes, self.study_info)
            self.best_guess_model = BestGuessModel(self.test_scenes, self.study_info)

        self.generate_model_lists()


def get_standard_preposition_parameters(train_scenes):
    model_study_info = StudyInfo("2019 study")
    # scene_list = model_study_info.scene_name_list
    preposition_models_dict = dict()

    features_to_remove = Configuration.object_specific_features.copy()
    # Get parameters for each preposition
    for p in PREPOSITION_LIST:
        M = GeneratePrepositionModelParameters(model_study_info, p, train_scenes,
                                               features_to_remove=features_to_remove)
        M.work_out_models()
        preposition_models_dict[p] = M

    return preposition_models_dict


def test_features(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    functional_features = ["location_control", "support"]
    m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/removed features",
                     BASIC_MODEL_SCORES_FOLDER + "plots/removed features", study_info_, number_runs=100, k=2,
                     features_to_test=functional_features, model_to_test_name=PrototypeModel.name)
    print("Test Features")
    m.validation()
    m.output()


def initial_test(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                     BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info_)
    print("Test on all scenes")
    m.test_all_scenes()


def test_models(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                     BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info_, number_runs=100, k=2, compare="y")
    print("Test Model k = 2")
    m.validation()
    m.output()

    # m = MultipleRuns(GenerateBasicModels, study_info_, number_runs=100, k=3, compare="y")
    # print("Test Model k = 3")
    # m.validation()
    # m.output()


def plot_all_csv(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                     BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info_)
    file = m.all_csv
    out_file = m.all_plot

    # self.plot_dataframe_bar_chart(self.all_dataframe,self.all_plot,"Preposition","Score","Scores Using All Data")
    m.plot_bar_from_csv(file, out_file, "Preposition", "Score", "Scores Using All Data")


def plot_kfold_csv(k, study_info_):
    """Summary
    
    Args:
        k (TYPE): Description
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/all features",
                     BASIC_MODEL_SCORES_FOLDER + "plots/all features", study_info_, number_runs=100, k=k)
    file = m.average_csv
    out_file = m.average_plot_pdf

    m.plot_bar_from_csv(file, out_file, "Preposition", "Score", m.average_plot_title)


def plot_feature_csv(k, study_info_):
    """Summary
    
    Args:
        k (TYPE): Description
        study_info_ (TYPE): Description
    """
    functional_features = ["location_control", "support"]
    m = MultipleRuns(GenerateBasicModels, BASIC_MODEL_SCORES_FOLDER + "tables/removed features",
                     BASIC_MODEL_SCORES_FOLDER + "plots/removed features", study_info_, number_runs=100, k=k,
                     features_to_test=functional_features, model_to_test_name=PrototypeModel.name)
    file = m.scores_tables_folder + "/functional_feature_analysis.csv"
    output_file = m.scores_plots_folder + "/ScoresWithRemovedFeatures.pdf"
    x_label = "Preposition"
    y_label = "Score"
    plot_title = "Average Scores With Removed Features. K = " + str(m.k) + " N = " + str(m.number_runs)

    m.plot_bar_from_csv(file, output_file, x_label, y_label, plot_title)


def main():
    """Summary
    
    Args:
        study_info_ (StudyInfo): Description
    """
    study_info = StudyInfo("2019 study")
    # plot_all_csv(study_info_)
    # 
    initial_test(study_info)
    test_models(study_info)
    test_features(study_info)
    pass


if __name__ == '__main__':


    main()

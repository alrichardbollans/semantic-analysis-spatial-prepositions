# Standard imports
import pandas as pd
import numpy as np
import math

# Ml modules

from sklearn.cluster import KMeans
from scipy.spatial.distance import minkowski
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Modules for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from baseline_model_testing import GeneratePrepositionModelParameters, PREPOSITION_LIST
from Analysis.data_import import Features
from data_import import Configuration, StudyInfo
from polysemy_analysis import GeneratePolysemeModels



class Cluster:
    """Summary

    Attributes:
        hr_means (TYPE): Description
        hr_means_csv (TYPE): Description
        instances (TYPE): Description
        instances_csv (TYPE): Description
        label (TYPE): Description
        mean_csv (TYPE): Description
        mean_series (TYPE): Description
        means (TYPE): Description
        preposition (TYPE): Description
        study_info (TYPE): Description
    """

    def __init__(self, study_info_, preposition, instances, label, alg_typ=None):
        """Summary

        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
            instances (TYPE): Description
            label (TYPE): Description
            alg_typ (None, optional): Description
        """
        self.study_info = study_info_
        self.preposition = preposition
        self.label = label
        self.instances = instances
        self.mean_series = self.instances.mean()
        # Convert series to appropriate dataframe
        self.means = pd.DataFrame({'values': self.mean_series.values})
        self.means = self.means.set_index(self.mean_series.index)
        self.means = self.means.transpose()

        feature_processer = self.study_info.feature_processor
        self.hr_means = feature_processer.convert_standard_df_to_normal(self.means)

        if alg_typ == "kmeans":
            base_folder = self.study_info.kmeans_folder
        elif alg_typ == "hry":
            base_folder = self.study_info.hry_folder
        else:
            print("Error: No cluster type given")

        self.mean_csv = base_folder + "cluster means/clusters-" + preposition + str(self.label) + ".csv"
        self.instances_csv = base_folder + "cluster instances/instances-" + preposition + str(self.label) + ".csv"

        self.hr_means_csv = base_folder + "cluster means/human readable/instances-" + preposition + str(
            self.label) + ".csv"

    def output(self):
        """Summary
        """
        # self.mean_series["cardinality"] = len(self.instances.index)
        self.means.to_csv(self.mean_csv)
        self.instances.to_csv(self.instances_csv)

        self.hr_means.to_csv(self.hr_means_csv)


class Clustering:
    """Summary

    Attributes:
        all_scenes (TYPE): Description
        cluster_centres_csv (TYPE): Description
        dendrogram_pdf (TYPE): Description
        elbow_pdf (TYPE): Description
        good_dataset (TYPE): Description
        good_instance_csv (TYPE): Description
        good_instance_features (TYPE): Description
        good_instances_all_features (TYPE): Description
        hr_good_instance_csv (TYPE): Description
        hr_good_instances (TYPE): Description
        initial_inertia_csv (TYPE): Description
        good_instances_to_cluster (TYPE): Description
        km_instances_to_cluster (TYPE): Description
        models (TYPE): Description
        possible_instances_all_features (TYPE): Description
        possible_instances_features (TYPE): Description
        preposition (TYPE): Description
        relation_weights (TYPE): Description
        sample_weights (TYPE): Description
        study_info (TYPE): Description
        typical_instances (TYPE): Description
    """

    # Number of clusters created by inspecting HAC dendograms
    cluster_numbers = {'on': 3, 'in': 2, 'against': 3, 'under': 3, 'over': 2}

    def __init__(self, study_info_, preposition, generated_polysemy_models=None):
        """Summary

        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
        """
        self.study_info = study_info_

        self.all_scenes = self.study_info.scene_name_list
        self.preposition = preposition

        self.generated_polysemy_models = generated_polysemy_models

        if generated_polysemy_models is not None:

            self.p_models_params = self.generated_polysemy_models.preposition_parameters_dict[preposition]
        else:
            self.p_models_params = GeneratePrepositionModelParameters(self.study_info, preposition,
                                                                      self.study_info.scene_name_list,
                                                                      features_to_remove=Configuration.object_specific_features)
        # All selected instances
        self.possible_instances_all_features = self.p_models_params.affAllFeatures
        self.possible_instances_features = self.p_models_params.affFeatures
        # Dataset containing 'good' instances
        self.good_dataset = self.p_models_params.good_dataset.copy()
        # Reindex df for later readability
        self.good_dataset = self.good_dataset.reset_index(drop=True)
        # All 'good' instances
        self.good_instances_all_features = self.p_models_params.goodAllFeatures.copy()

        self.good_instance_features = self.p_models_params.goodFeatures.copy()

        self.typical_instances = self.p_models_params.typical_features

        # self.p_models_params.all_feature_weights()
        # self.feature_weights= self.p_models_params.read_all_feature_weights()
        self.relation_weights = self.p_models_params.read_regression_weights()

        self.cluster_centres_csv = self.study_info.kmeans_folder + "cluster centres/clusters-" + preposition + ".csv"
        self.dendrogram_pdf = self.study_info.hry_folder + "figures/dendrogram/dendrogram-" + preposition + ".pdf"
        self.elbow_pdf = self.study_info.kmeans_folder + "figures/elbow/" + preposition + ".pdf"
        self.initial_inertia_csv = self.study_info.kmeans_folder + "initial inertia/initial_inertias.csv"
        # The dataframe we use for clustering
        self.good_instances_to_cluster = self.good_instance_features.copy()
        self.km_instances_to_cluster = self.possible_instances_features.copy()
        # Samples are weighted by selection ratio
        self.sample_weights = self.p_models_params.aff_dataset[
            self.p_models_params.ratio_feature_name]  # self.good_dataset[self.p_models_params.ratio_feature_name]

        self.km_instances_to_cluster.to_csv("test/test outputs/clustering test/" + preposition + "km.csv")

    def custom_metric(self, u, v):
        """Summary

        Args:
            u (TYPE): Description
            v (TYPE): Description

        Returns:
            TYPE: Description
        """
        # weighted euclidean distance. Also weight by instances somehow?
        return minkowski(u, v, p=2, w=self.relation_weights.values)

    def output_good_instances(self):
        # Output good instances to read
        good_instance_csv = self.study_info.cluster_data_folder + "good preposition instances/good instances - " + self.preposition + ".csv"
        self.good_instances_to_cluster.to_csv(good_instance_csv)

        feature_processer = Features(self.study_info.input_feature_csv, self.study_info.feature_output_csv,
                                     self.study_info.means_output_path,
                                     self.study_info.std_output_path,
                                     self.study_info.human_readable_feature_output_csv)
        hr_good_instance_csv = self.study_info.cluster_data_folder + "good preposition instances/human readable/good instances - " + self.preposition + ".csv"

        hr_good_instances = feature_processer.convert_standard_df_to_normal(self.good_instances_to_cluster)

        hr_good_instances.to_csv(hr_good_instance_csv)

    def work_out_hierarchy_model(self):
        """Summary
        """
        instances = self.good_instances_to_cluster
        Z = linkage(instances, method='single', optimal_ordering=True)  # ,metric=self.custom_metric

        fig = plt.figure()
        fig.canvas.set_window_title(self.preposition)
        # fig.suptitle("Dendrogram for '"+ self.preposition+"'", fontsize=20)

        # factor = self.color_threshold[self.preposition]
        if self.preposition == "on":
            thresh = 0.65 * max(Z[:, 2])
        else:
            thresh = 0.7 * max(Z[:, 2])  # Default threshold
        dendrogram(Z, color_threshold=thresh)

        plt.xlabel('Preposition Instances')
        plt.ylabel('Distances between clusters')
        plt.savefig(self.dendrogram_pdf, bbox_inches='tight')

        # Form flat clusters based on threshold
        clusters = fcluster(Z, criterion='distance', t=thresh)

        done_clusters = []
        for c in clusters:
            if c not in done_clusters:
                done_clusters.append(c)
                cluster_instances_index = []
                for i in range(len(clusters)):
                    if c == clusters[i]:
                        cluster_instances_index.append(i)
                cluster_instances = instances.iloc[cluster_instances_index, :]
                cluster = Cluster(self.study_info, self.preposition, cluster_instances, c, alg_typ="hry")
                cluster.output()
        print(("Number of clusters: " + str(len(done_clusters))))

    def work_out_kmeans_model(self, k):
        """Summary

        Args:
            k (TYPE): Description

        Returns:
            TYPE: Description
        """
        number_clusters = k

        # set random state for repeatability
        km = KMeans(
            n_clusters=number_clusters, random_state=0

        )
        km.fit(self.km_instances_to_cluster, sample_weight=self.sample_weights)

        return km

    def output_cluster_info(self, km):
        """Summary

        Args:
            km (TYPE): Description
        """
        out = dict()

        for i in range(len(km.cluster_centers_)):
            out["cluster_" + str(i)] = km.cluster_centers_[i]

        df = pd.DataFrame(out, self.p_models_params.feature_keys)
        print(self.preposition)
        print(df)

        df.to_csv(self.cluster_centres_csv)

        k = self.cluster_numbers[self.preposition]
        for i in range(0, k):
            instances = self.km_instances_to_cluster[km.labels_ == i]
            cluster = Cluster(self.study_info, self.preposition, instances, i, alg_typ="kmeans")
            cluster.output()

    def output_expected_kmeans_model(self):
        """Summary
        """
        k = self.cluster_numbers[self.preposition]
        kmodel = self.work_out_kmeans_model(k)

        self.output_cluster_info(kmodel)

    def check_inertia_calculation(self):
        """Summary
        """
        # Checks inertia calculation agrees with KMeans method
        km = self.work_out_kmeans_model(4)
        print("Our inertia")
        i = self.calculate_inertia_from_centres(km.cluster_centers_)
        print(i)
        print("proper inertia")
        print((km.inertia_))

    def distance_to_centre_squared(self, point, centre):
        """Summary

        Args:
            point (TYPE): Description
            centre (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Subtract arrays point wise
        point = np.subtract(point, centre)
        # Square pointwise
        point = np.square(point)
        # Sum
        summ = np.sum(point)

        # Square root to get distance
        distance = math.sqrt(summ)

        d2 = distance * distance

        return d2

    def calculate_inertia_from_centres(self, centres):
        """Summary
        Given a set of centres, calculate the inertia when clustering the km_instances_to_cluster
        Args:
            centres (TYPE): Description

        Returns:
            TYPE: Description
        """
        total_sum = 0
        # As per the kmeans source code, sample weights are scaled so average weight is 1
        weight_scaling_factor = len(self.sample_weights) / self.sample_weights.sum()

        for index, row in self.km_instances_to_cluster.iterrows():

            sample = row.values
            closest_distance = -1
            closest_centre = 0
            for centre in centres:
                if not np.isnan(centre[0]):
                    d = self.distance_to_centre_squared(sample, centre)
                    if closest_distance == -1:
                        closest_distance = d
                        closest_centre = centre
                    elif d < closest_distance:
                        closest_distance = d
                        closest_centre = centre

            weight = self.sample_weights[index]
            normalised_weight = weight * weight_scaling_factor
            weighted_distance = closest_distance * normalised_weight
            total_sum += weighted_distance

        return total_sum

    def calculate_polysemes_inertia(self, polysemes):
        """Summary
        Given a set of polysemes, take their mean instance values to create a list of centres.
        Then use the centres to calculate inertia.
        Args:
            polysemes (list): List of polysemes

        Returns:
            TYPE: Description
        """
        init = []
        centres = []
        for polyseme in polysemes:
            # df = pd.read_csv(polyseme.mean_csv, index_col=0, names=["feature", "value"])
            # print(df)
            # centres.append(df["value"].values)
            # print(centres)
            centres.append(polyseme.preposition_models.affFeatures.mean())

        i = self.calculate_inertia_from_centres(centres)
        print(self.preposition)
        print(("Number of clusters:" + str(len(centres))))
        return i

    def plot_elbow_polyseme_inertia(self):
        """Summary
        """

        polysemes = self.generated_polysemy_models.non_shared.polyseme_dict[self.preposition]
        polysemes_inertia = self.calculate_polysemes_inertia(polysemes)

        refined_polysemes = self.generated_polysemy_models.refined.polyseme_dict[self.preposition]
        refined_polysemes_inertia = self.calculate_polysemes_inertia(refined_polysemes)

        inertias = []
        K = list(range(1, 10))
        for k in K:
            kmeanModel = self.work_out_kmeans_model(k)

            inertias.append(kmeanModel.inertia_)

        fig, axes = plt.subplots()

        # Plot inertia from model
        axes.plot([len(polysemes)], [polysemes_inertia], markersize=15, markeredgewidth=3, linestyle='None',
                  marker=(5, 2), label="Polysemes")

        # Plot inertia from refined model
        axes.plot([len(refined_polysemes)], [refined_polysemes_inertia], markersize=15, markeredgewidth=3,
                  linestyle='None',
                  marker='+', label="Refined Polysemes")

        # Plot the elbow
        axes.plot(K, inertias, 'bx-', label="K-Means")

        axes.set_xlabel('Number of Clusters')
        axes.set_ylabel('Inertia')
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # axes.set_title('Inertia from Kmeans Clusters - ' + self.preposition, pad=10)

        plt.legend(loc='upper right')

        plt.savefig(self.elbow_pdf, bbox_inches='tight')

    # plt.close()

    def output_initial_inertia(self):
        """Summary
        """
        kmeanModel = self.work_out_kmeans_model(1)
        inertia = kmeanModel.inertia_

        normalised_inertia = inertia / len(self.km_instances_to_cluster.index)
        new_csv = False
        try:

            in_df = pd.read_csv(self.initial_inertia_csv, index_col=0)

        except Exception as e:
            in_df = pd.DataFrame(columns=['preposition', 'inertia', 'divided by number of instances'])
            # print("unsusccefully read")
            new_csv = True

        finally:

            row_index_in_df = in_df[in_df['preposition'] == self.preposition].index.tolist()

            if len(row_index_in_df) == 0:
                in_df = in_df.append({'preposition': self.preposition, 'inertia': inertia,
                                      'divided by number of instances': normalised_inertia}, ignore_index=True)
            else:

                in_df.at[row_index_in_df[0], 'inertia'] = inertia
                in_df.at[row_index_in_df[0], 'divided by number of instances'] = normalised_inertia

            in_df.to_csv(self.initial_inertia_csv)


def output_clustering_info(study_info_):
    """Summary
    :param study_info_:

    Args:
        study_info_ (TYPE): Description
    """
    all_scenes = study_info_.scene_name_list
    generated_polysemy_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_)
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['legend.fontsize'] = 12
    for preposition in PREPOSITION_LIST:
        c = Clustering(study_info_, preposition, generated_polysemy_models=generated_polysemy_models)

        c.plot_elbow_polyseme_inertia()
        c.output_initial_inertia()

    # for preposition in POLYSEMOUS_PREPOSITIONS:
    #     c = Clustering(study_info_, preposition)
    #     c.plot_elbow_polyseme_inertia()
    #     c.output_initial_inertia()
    #     c.output_expected_kmeans_model()


def work_out_all_hry_clusters(study_info_):
    """Summary
    :param study_info_:

    Args:
        study_info_ (TYPE): Description
    """
    # all_scenes = study_info_.scene_name_list
    # generated_polysemy_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_,
    # preserve_empty_polysemes=True)
    print("Working out hry clusters")
    for preposition in PREPOSITION_LIST:
        print(preposition)
        c = Clustering(study_info_, preposition)
        c.work_out_hierarchy_model()
        c.output_good_instances()


def main(study_info_):
    """Un/comment functions to run tests and outputs

    Args:
        study_info_ (TYPE): Description
        :param study_info_:

    Deleted Parameters:
        constraint_dict (TYPE): Description
    """

    # work_out_all_hry_clusters(study_info_)

    mpl.rcParams['axes.titlesize'] = 'large'
    mpl.rcParams['axes.labelsize'] = 'large'
    output_clustering_info(study_info_)
    # work_out_all_hry_clusters(study_info_)


if __name__ == '__main__':
    study_info = StudyInfo("2019 study")

    main(study_info)

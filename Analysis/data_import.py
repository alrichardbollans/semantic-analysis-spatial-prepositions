"""Summary
Methods and classes for loading the data, related to data format.
"""
import inspect
import os
import csv
import itertools

import pandas as pd


def get_git_project_directory():
    """
    Returns:
        str: Path to base unity project folder
    """
    unity_folder_name = "Unity Projects"
    repo_name = "spatial-preposition-annotation-tool-unity3d"
    current_directory = os.getcwd()
    user_home = os.path.expanduser("~")

    if os.path.basename(current_directory) == repo_name:
        return current_directory
    if os.path.basename(os.path.dirname(current_directory)) == repo_name:
        return os.path.dirname(current_directory)
    return user_home + "/Dropbox/" + unity_folder_name + "/" + repo_name


def clean_name(object_name):
    """Summary
    
    Args:
        object_name (string): Description
    
    Returns:
        string: Description
    """
    if "." in object_name:
        new_clean_name = object_name[: object_name.find(".")]
    elif "_" in object_name:
        new_clean_name = object_name[: object_name.find("_")]
    else:
        new_clean_name = object_name
    return new_clean_name.lower()


def remove_dot_unity(scene_name):
    """Summary
    
    Args:
        scene_name (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    new_clean_name = scene_name
    if ".unity" in scene_name:
        new_clean_name = scene_name[: scene_name.find(".unity")]
    return new_clean_name


class MyScene:
    """Summary
    
    Attributes:
        example_scenes (list): Description
        mesh_objects (TYPE): Description
        name (TYPE): Description
        selectable_objects (TYPE): Description
        study_scene (TYPE): Description
        unselectable_scene_objects (list): Description
    """

    # Class to store info about scenes
    unselectable_scene_objects = ["wall", "floor", "ceiling"]
    # This is given in MyScene class in Unity

    example_scenes = [
        "example",
        "finish",
        "instruction",
        "template",
        "main",
        "player",
        "screen",
        "test",
    ]

    def __init__(self, name, mesh_objects):
        """Summary
        
        Args:
            name (TYPE): Description
            mesh_objects (TYPE): Description
        """
        self.name = name
        self.mesh_objects = mesh_objects
        self.selectable_objects = self.get_selectable_objects()  #
        # Bool. Is this scene used for study_name?
        self.study_scene = self.study_scene_check()

    def get_selectable_objects(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        for obj in self.mesh_objects:
            if not any(x in obj for x in self.unselectable_scene_objects):
                out.append(obj)
        return out

    def get_all_configs(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []

        x = list(itertools.permutations(self.selectable_objects, 2))

        return x

    def study_scene_check(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        if any(x in self.name for x in self.example_scenes):
            return False
            # print(self.name)
        else:
            return True
            # print(self.name)


class StudyInfo:
    """Summary
    
    Class containing basic info related to data collection and directory structure
    
    Attributes:
        a_index (dict): Description
        analysis_folder_name (str): Description
        base_collected_data_folder_name (str): Description
        base_feature_data_folder_name (str): Description
        base_polysemy_folder (TYPE): Description
        base_stats_folder_name (str): Description
        cluster_data_folder (TYPE): Description
        comp_annotations_name (str): Description
        comp_task (str): Description
        comparative_abbreviations (TYPE): Description
        comparative_preposition_list (TYPE): Description
        config_list (TYPE): Description
        config_ratio_folder (TYPE): Description
        constraint_csv (TYPE): Description
        data_folder (TYPE): Description
        feature_keys (list): Description
        feature_output_csv (TYPE): Description
        feature_output_folder (TYPE): Description
        hry_folder (TYPE): Description
        human_readable_feature_output_csv (TYPE): Description
        input_feature_csv (TYPE): Description
        input_feature_data_folder (TYPE): Description
        kmeans_folder (TYPE): Description
        model_info_folder (TYPE): Description
        name (TYPE): Description
        polyseme_data_folder (TYPE): Description
        polysemy_score_folder (TYPE): Description
        PREPOSITION_LIST (list): Description
        project_path (TYPE): Description
        raw_annotation_csv (TYPE): Description
        raw_user_csv (TYPE): Description
        feature_keys (list): Description
        scene_info_csv_file (TYPE): Description
        scene_info_filename (str): Description
        sem_annotations_name (str): Description
        semantic_abbreviations (TYPE): Description
        semantic_preposition_list (TYPE): Description
        stats_folder (TYPE): Description
        study_list (list): Description
        sv_task (str): Description
        svmod_annotations_name (str): Description
        svmod_task (str): Description
        typ_a_index (dict): Description
        typ_annotations_name (str): Description
        typ_task (str): Description
        u_index (TYPE): Description
        unity_project_folder_name (str): Description
    
    Deleted Attributes:
        study_name (TYPE): Description
    """

    unity_project_folder_name = "Unity3D Annotation Environment"
    analysis_folder_name = "Analysis"
    base_feature_data_folder_name = "Scene Data"

    base_collected_data_folder_name = "collected data"
    sem_annotations_name = "clean semantic annotation list.csv"
    svmod_annotations_name = "clean svmod annotation list.csv"
    comp_annotations_name = "clean comparative annotation list.csv"
    typ_annotations_name = "clean typicality annotation list.csv"

    base_stats_folder_name = "stats"

    # Prepositions
    preposition_list = [
        "in",
        "inside",
        "against",
        "on",
        "on top of",
        "under",
        "below",
        "over",
        "above",
    ]

    semantic_preposition_list = preposition_list

    comparative_preposition_list = preposition_list

    # Task abbreviations
    sv_task = "sv"
    comp_task = "comp"
    typ_task = "typ"
    svmod_task = "sv_mod"
    semantic_abbreviations = [sv_task, svmod_task]

    comparative_abbreviations = [comp_task]

    # Dictionary giving the index of each value in raw annotations

    a_index = {
        "id": 0,
        "userid": 1,
        "time": 2,
        "figure": 3,
        "ground": 4,
        "task": 5,
        "scene": 6,
        "preposition": 7,
        "prepositions": 8,
        "cam_rot": 9,
        "cam_loc": 10,
    }
    # indexs for typ task
    typ_a_index = {
        "id": 0,
        "userid": 1,
        "time": 2,
        "c1": 3,
        "c2": 4,
        "task": 5,
        "selection": 6,
        "preposition": 7,
    }

    # Dictionary giving index of values in raw userlist

    u_index = {
        "user_id": 0,
        "time": 1,
        "native": 2
    }
    study_list = ["2019 study", "2020 study"]
    project_path = get_git_project_directory()

    scene_info_filename = "scene_info.csv"

    def __init__(self, study_name):
        """Summary
        
        Args:
            study_name (TYPE): Description
        """
        # Abbreviation for files and folders
        self.name = study_name

        # paths and filenames

        self.input_feature_data_folder = (

                self.name
                + "/"
                + self.base_feature_data_folder_name
        )

        self.input_feature_csv = self.input_feature_data_folder + "/relations.csv"

        self.feature_output_folder = self.name + "/" + "feature values"
        # Path for outputting feature values
        self.feature_output_csv = (
                self.feature_output_folder + "/standardised_values.csv"
        )
        # Path for outputting human-readable feature values
        self.human_readable_feature_output_csv = (
                self.feature_output_folder + "/human_readable_values.csv"
        )

        self.scene_info_csv_file = self.input_feature_data_folder + "/" + self.scene_info_filename

        self.data_folder = self.name + "/" + self.base_collected_data_folder_name
        self.raw_user_csv = self.data_folder + "/" + "userlist.csv"
        self.clean_user_csv = self.data_folder + "/" + "clean_users.csv"

        self.raw_annotation_csv = self.data_folder + "/" + "annotationlist.csv"

        self.stats_folder = self.name + "/" + self.base_stats_folder_name

        self.config_ratio_folder = self.name + "/" + 'preposition data/'

        self.constraint_csv = self.name + "/constraint data/constraints.csv"

        self.model_info_folder = self.name + "/model info"

        self.base_polysemy_folder = self.name + "/polysemy/"
        self.polyseme_data_folder = self.base_polysemy_folder + 'polyseme data/'
        self.cluster_data_folder = self.base_polysemy_folder + 'clustering/'
        self.kmeans_folder = self.cluster_data_folder + 'kmeans/'
        self.hry_folder = self.cluster_data_folder + 'hry/'
        self.polysemy_score_folder = self.base_polysemy_folder + 'scores/'

        # Useful lists
        self.scene_list, self.scene_name_list = self.get_scenes()
        self.user_name_list = self.get_user_names()

        self.all_feature_keys = []

        self.config_list = self.load_all_configs()

        self.means_output_path = self.feature_output_folder + "/feature_means.csv"
        self.std_output_path = self.feature_output_folder + "/feature_stds.csv"
        self.feature_processor = Features(self.input_feature_csv, self.feature_output_csv, self.means_output_path,
                                          self.std_output_path, self.human_readable_feature_output_csv)

    def load_all_configs(self):
        """Summary
        
        Loads a list of all configurations from csv containing feature values.
        May be better to get this from annotation list to minimise size of list.
        
        Returns:
            list: list of configurations.
        
        Deleted Parameters:
            path (None, optional): Description
            feature_path (str): path to feature values
        """
        try:
            with open(self.feature_output_csv, "r") as f:
                reader = csv.reader(f)  # create a 'csv reader' from the file object
                geom_relations = list(reader)  # create a list from the reader

                for title in geom_relations[0][3:]:
                    self.all_feature_keys.append(title)

                geom_relations.pop(0)
                config_list = []

                for row in geom_relations:
                    c1 = Configuration(row[0], row[1], row[2], self)
                    config_list.append(c1)

            return config_list
        # If features haven't yet been preprocessed the feature_output csv won't exist
        except Exception as e:

            print(e)

    def config_ratio_csv(self, filetag, preposition):
        """Summary
        
        Args:
            filetag (TYPE): Description
            preposition (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return self.config_ratio_folder + filetag + '-ratio-list' + preposition + ' .csv'

    def get_user_names(self):
        try:
            sem_annotations_csv = self.data_folder + "/" + self.sem_annotations_name
            names_out = []
            with open(sem_annotations_csv, "r") as file:
                reader = csv.reader(file)
                datalist = list(reader)

                for ann in datalist[1:]:
                    user = ann[1]
                    if user not in names_out:
                        names_out.append(user)

            return names_out
        except FileNotFoundError:
            return None

    def get_scenes(self):
        """Summary
        
        Returns:
            list: Description
        """
        scenes_out = []
        names_out = []
        with open(self.scene_info_csv_file, "r") as file:
            reader = csv.reader(file)
            datalist = list(reader)

            for scene_info in datalist:
                mesh_objects = []
                for i in range(1, len(scene_info)):
                    mesh_objects.append(scene_info[i])
                if len(mesh_objects) != 0 and ".unity" in scene_info[0]:
                    s_name = remove_dot_unity(scene_info[0])
                    my_scene = MyScene(s_name, mesh_objects)

                    if my_scene.study_scene:
                        scenes_out.append(my_scene)
                        names_out.append(my_scene.name)

        return scenes_out, names_out
    # @staticmethod
    # def get_scene_list():
    #     """Summary

    #     Returns:
    #         TYPE: Description
    #     """
    #     scene_list = []

    #     s = SceneInfo()
    #     for scene in s.scene_list:
    #         scene_list.append(scene.name)
    #     return scene_list


class SimpleConfiguration:
    """Summary
    Simple Configuration class which only has scene, figure and ground names.
    Avoids loading study info and feature values.
    
    Attributes:
        figure (TYPE): Description
        ground (TYPE): Description
        scene (TYPE): Description
    
    """

    def __init__(self, scene, figure, ground):
        """Summary
        
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
        """
        self.scene = scene
        self.ground = ground
        self.figure = figure

    def configuration_match(self, instance):
        """Summary
        Checks if object matches the current configuration. Object may be anything with scene, figure and grounds attribute.
        
        Args:
            instance (TYPE): Description
        
        Returns:
            bool: Description
        """
        if (
                self.scene == instance.scene
                and self.figure == instance.figure
                and self.ground == instance.ground
        ):
            return True
        else:
            return False

    def number_of_selections_from_instancelist(self, preposition, instancelist):
        """Summary
        Checks how many instances there are of the configuration being assigned with the preposition.
        From an instance list where annotations are separated by preposition.
        
        Args:
            preposition (TYPE): Description
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        counter = 0
        for i in instancelist:
            if self.configuration_match(i) and i.preposition == preposition:
                counter += 1
        return counter

    def number_of_selections_from_annotationlist(self, preposition, annotationlist):
        """Summary
        
        Args:
            preposition (TYPE): Description
            annotationlist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        counter = 0
        for an in annotationlist:

            if self.configuration_match(an) and preposition in an.prepositions:
                counter += 1
        return counter

    def number_of_tests(self, simple_config_list):
        """Summary
        Finds how many times this configuration appears in a list of configurations.
        
        Args:
            simple_config_list (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Need to use annotation list here as instances are separated by preposition
        counter = 0
        for c in simple_config_list:

            if self.configuration_match(c):
                counter += 1
        # print(counter)
        return counter

    def ratio_semantic_selections(self, preposition, annotationlist, instancelist):
        """Summary
        
        Args:
            preposition (TYPE): Description
            annotationlist (TYPE): Description
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        t = float(self.number_of_tests(annotationlist))
        s = float(self.number_of_selections_from_instancelist(preposition, instancelist))

        if t != 0:
            return s / t
        else:
            return 0

    def print_info(self):
        """Summary
        """
        print(("Scene = " + self.scene))
        print(("Figure = " + self.figure))
        print(("Ground = " + self.ground))

    def string_for_cattyp_table(self):
        return "[" + self.figure + "," + self.ground + "]"


class Configuration(SimpleConfiguration):
    """class which is used to read feature values of configurations.
    
    Attributes:
        ground_property_features (list): Description
        full_row (TYPE): Description
        relations_row (list): Description
        row (list): Description
        scene (TYPE): Description
        set_of_features (dict): Description
        study (TYPE): Description
    
    Deleted Attributes:
        output_path (TYPE): Description
        input_feature_csv (TYPE): Description
        data_path (TYPE): Description
        all_feature_keys (list): Description
        feature_keys (list): Description
        figure (TYPE): Description
        ground (TYPE): Description
    """

    # Lots of this could be done with pandas. Doh :/

    object_specific_features = ["ground_lightsource", "ground_container", "ground_verticality", "figure_lightsource",
                                "figure_container","size_ratio"]

    def __init__(self, scene, figure, ground, study):
        """Summary
        
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
            study (TYPE): Description
        """
        SimpleConfiguration.__init__(self, scene, figure, ground)
        self.study = study
        # Dictionary of features and values
        self.set_of_features = {}

        # Row of feature values for outputing to csv
        self.row = []
        # Row beginning with names
        self.full_row = [self.scene, self.figure, self.ground]

        if self.figure != "none":
            self.load_from_csv()

    def __str__(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return (
                "["
                + str(self.scene)
                + ","
                + str(self.figure)
                + ","
                + str(self.ground)
                + "]"
        )

    def load_from_csv(self):
        """Summary
        
        Deleted Parameters:
            path (None, optional): Description
        """

        with open(self.study.feature_output_csv, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            geom_relations = list(reader)  # create a list from the reader

            for relation in geom_relations:
                if (
                        self.scene == relation[0]
                        and self.figure == relation[1]
                        and self.ground == relation[2]
                ):
                    # print(geom_relations.index(relation))
                    for key in self.study.all_feature_keys:
                        value_index = self.study.all_feature_keys.index(key) + 3
                        if relation[value_index] != "?":
                            self.set_of_features[key] = float(
                                relation[value_index]
                            )
                        else:
                            self.set_of_features[key] = "?"

                        value = self.set_of_features[key]
                        setattr(self, key, value)
                        self.row.append(value)
                        self.full_row.append(value)


class Features:
    """Summary
    Provides methods for standardising feature values and converting between raw values and standardised values.
    Attributes:
        dataset (TYPE): Description
        features_to_remove (TYPE): Description
        human_readable_path (TYPE): Description
        means (TYPE): Description
        output_path (TYPE): Description
        input_feature_csv (TYPE): Description
        stds (TYPE): Description
    """

    # Features given in relation.csv that aren't needed for analysis
    features_to_remove = ["horizontal_distance_normalized", "shortest_distance_normalized",
                          "location_control_-x", "location_control_-z", "location_control_x",
                          "location_control_z", "figure_volume", "ground_volume",
                          "horizontal_projection_overlap_proportion"]
    # Number of columns at start of each row which do not contain numeric values.
    number_of_non_value_columns = 3

    def __init__(self, input_feature_csv, output_feature_csv, output_mean_csv, output_std_csv,
                 output_human_readable_feature_csv):
        """Summary

        Args:
            study (TYPE): Description
        """

        # Get path for calculated features
        self.input_feature_csv = input_feature_csv
        self.output_path = output_feature_csv
        self.means_output_path = output_mean_csv
        self.std_output_path = output_std_csv
        self.human_readable_path = output_human_readable_feature_csv

        self.dataset = pd.read_csv(self.input_feature_csv)
        # Remove values where fig=ground

        self.dataset = self.dataset[self.dataset.Figure != self.dataset.Ground]

        # Calculate location control and append column
        location_control_parts = ["location_control_x", "location_control_-x", "location_control_-z",
                                  "location_control_z"]

        self.dataset["location_control"] = self.dataset[location_control_parts].sum(axis=1) / 4

        # Remove unwanted columns
        self.dataset = self.dataset.drop(self.features_to_remove, axis=1)
        # Output cleaned dataset
        self.dataset.to_csv(self.human_readable_path, index=False)
        # Calculate means and standard deviations
        # Can be used to convert standardised values back to real values
        # Note that different pandas versions will give slightly different float values, which has knock on effects
        # E.g. some configurations are then classed as possible polyseme instances which weren't previously
        # Here we use pandas version 1.0.3
        self.means = self.dataset.iloc[:, self.number_of_non_value_columns:].mean()
        self.stds = self.dataset.iloc[:, self.number_of_non_value_columns:].std()

    def standardise_values(self):
        """Summary

        Returns:
            TYPE: Description
        """
        new_dataframe = self.dataset.copy()

        new_dataframe.iloc[:, self.number_of_non_value_columns:] = new_dataframe.iloc[:,
                                                                   self.number_of_non_value_columns:] - self.means
        new_dataframe.iloc[:, self.number_of_non_value_columns:] = new_dataframe.iloc[:,
                                                                   self.number_of_non_value_columns:] / (self.stds)

        return new_dataframe

    def write_new(self, dataframe):
        """Summary

        Args:
            dataframe (TYPE): Description
        """
        dataframe.to_csv(self.output_path, index=False)

    def write_mean_std(self):
        """Summary
        """
        self.means.to_csv(self.means_output_path)
        self.stds.to_csv(self.std_output_path)

    def convert_standardised_value_to_normal(self, feature, value):
        """Summary

        Args:
            feature (TYPE): Description
            value (TYPE): Description

        Returns:
            TYPE: Description
        """
        mean = self.means[feature]
        std = self.stds[feature]

        x = (value * std) + mean

        return x

    def convert_normal_value_to_standardised(self, feature, value):
        """Summary

        Args:
            feature (TYPE): Description
            value (TYPE): Description

        Returns:
            TYPE: Description
        """
        mean = self.means[feature]
        std = self.stds[feature]

        x = (value - mean) / std

        return x

    def get_max_value(self, feature):
        """Summary

        Args:
            feature (TYPE): Description

        Returns:
            TYPE: Description
        """
        m = self.dataset[feature].max()
        return m

    def get_min_value(self, feature):
        """Summary

        Args:
            feature (TYPE): Description

        Returns:
            TYPE: Description
        """
        m = self.dataset[feature].min()
        return m

    def convert_standard_df_to_normal(self, df):
        """Summary

        Args:
            df (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Need to be careful assigning new dataframe variables this way
        # As it returns a reference to the initial dataframe UNLESS copy() is used
        new_dataframe = df.copy()
        for column in new_dataframe:
            mean = self.means[column]
            std = self.stds[column]

            new_dataframe[column] = (new_dataframe[column] * std) + mean

        return new_dataframe

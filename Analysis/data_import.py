"""Summary
Methods and classes for loading the data, related to data format.
"""

import os
import csv
import itertools


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
        preposition_list (list): Description
        project_path (TYPE): Description
        raw_annotation_csv (TYPE): Description
        raw_user_csv (TYPE): Description
        relation_keys (list): Description
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
        self.scene_list, self.scene_name_list = self.get_scenes()

        self.config_list = self.load_all_configs()

        self.data_folder = self.name + "/" + self.base_collected_data_folder_name
        self.raw_user_csv = self.data_folder + "/" + "userlist.csv"

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

    def load_all_configs(self):
        """Summary
        
        Loads a list of all configurations.
        
        Returns:
            list: list of configurations.
        
        Deleted Parameters:
            path (None, optional): Description
            feature_path (str): path to feature values
        """

        with open(self.feature_output_csv, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            geom_relations = list(reader)  # create a list from the reader

            self.feature_keys = []
            self.relation_keys = []
            for title in geom_relations[0][3:]:
                self.feature_keys.append(title)
                if title not in Configuration.context_features:
                    self.relation_keys.append(title)

            geom_relations.pop(0)
            config_list = []

            for row in geom_relations:
                c1 = Configuration(row[0], row[1], row[2], self)
                config_list.append(c1)

        return config_list

    def config_ratio_csv(self, filetag, preposition):
        """Summary
        
        Args:
            filetag (TYPE): Description
            preposition (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return self.config_ratio_folder + filetag + '-ratio-list' + preposition + ' .csv'

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


class Configuration:
    """class which is used to read feature values
    
    Attributes:
        context_features (list): Description
        figure (TYPE): Description
        full_row (TYPE): Description
        ground (TYPE): Description
        relations_row (list): Description
        row (list): Description
        scene (TYPE): Description
        set_of_features (dict): Description
        study (TYPE): Description
    
    Deleted Attributes:
        output_path (TYPE): Description
        input_feature_csv (TYPE): Description
        data_path (TYPE): Description
        feature_keys (list): Description
        relation_keys (list): Description
    """

    # Lots of this could be done with pandas. Doh :/

    # input_feature_csv = StudyInfo.feature_output_csv
    # output_path = input_feature_csv

    # additional_features = ["location_control"]
    # Location control is the average of the two more basic measures

    context_features = ["ground_lightsource", "ground_container", "ground_verticality"]

    def __init__(self, scene, figure, ground, study):
        """Summary
        
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
            study (TYPE): Description
        """
        self.study = study
        self.scene = scene
        self.figure = figure
        self.ground = ground
        # Dictionary of features and values
        self.set_of_features = {}

        # Row of feature values for outputing to csv
        self.row = []
        # Row beginning with names
        self.full_row = [self.scene, self.figure, self.ground]
        # Row without context features
        self.relations_row = []
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
                    for key in self.study.feature_keys:
                        if relation[self.study.feature_keys.index(key) + 3] != "?":
                            self.set_of_features[key] = float(
                                relation[self.study.feature_keys.index(key) + 3]
                            )
                        else:
                            self.set_of_features[key] = "?"

                        value = self.set_of_features[key]
                        setattr(self, key, value)
                        self.row.append(value)
                        self.full_row.append(value)
                        if key in self.study.relation_keys:
                            self.relations_row.append(value)

    def configuration_match(self, instance):
        """Summary
        
        Args:
            instance (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if (
                self.scene == instance.scene
                and self.figure == instance.figure
                and self.ground == instance.ground
        ):
            return True
        else:
            return False

    def number_of_selections(self, preposition, instancelist):
        """Summary
        
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
            if self.annotation_row_match(an) and preposition in an.prepositions:
                counter += 1
        return counter

    def number_of_tests(self, annotationlist):
        """Summary
        
        Args:
            annotationlist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Need to use annotation list here as instances are separated by preposition
        counter = 0
        for an in annotationlist:

            if self.annotation_row_match(an):
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
        s = float(self.number_of_selections(preposition, instancelist))

        if t != 0:
            return s / t
        else:
            return 0

    def annotation_row_match(self, row):
        """Summary
        
        Args:
            row (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # If configuration matches with annotation in raw data list
        if self.scene == row[3] and self.figure == row[5] and self.ground == row[6]:
            return True
        else:
            return False

    def config_row_match(self, value):
        """Summary
        
        Args:
            value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if (
                self.scene == value[0]
                and self.figure == value[1]
                and self.ground == value[2]
        ):
            return True
        else:
            return False

    def print_info(self):
        """Summary
        """
        print(("Scene = " + self.scene))
        print(("Figure = " + self.figure))
        print(("Ground = " + self.ground))

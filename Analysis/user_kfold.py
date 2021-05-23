# So far any kfold testing has used train-test *scenes* and not considered splitting participants
# This file runs tests which also split participants
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_model_testing import GeneratePrepositionModelParameters, PREPOSITION_LIST, PrototypeModel, ExemplarModel, \
    CSModel, ProximityModel, SimpleModel, BestGuessModel
from classes import Instance, CompInstance, Comparison
from compile_instances import SemanticCollection, ComparativeCollection
from data_import import SimpleConfiguration, StudyInfo, Configuration
from Analysis.performance_test_functions import MultipleRuns
from process_data import SemanticAnnotation, ComparativeAnnotation

TEST_FOLDER = 'extra thesis results/user kfold'


def new_config_ratio_csv(preposition, study_info):
    filetag = 'semantic'
    new_csv = TEST_FOLDER + '/' + study_info.config_ratio_csv(filetag, preposition)
    return new_csv


def new_constraint_csv(study_info):
    new_csv = TEST_FOLDER + '/' + study_info.constraint_csv
    return new_csv


# First new csvs need writing

# class NewSemanticCollection(SemanticCollection):
#
#     def get_tested_users_dict(self):
#         config_list = self.study_info.config_list
#         user_dict = {}
#         for c in config_list:
#             user_dict[c] = []
#         filename = self.study_info.data_folder + "/" + self.study_info.sem_annotations_name
#
#         with open(filename, "r") as f:
#             reader = csv.reader(f)  # create a 'csv reader' from the file object
#             annotationlist = list(reader)  # create a list from the reader
#
#             annotationlist.pop(0)  # removes first line of data list which is headings
#
#             for clean_annotation in annotationlist:
#                 an_id, clean_user_id, task, scene, prepositions, figure, ground, time = SemanticAnnotation.retrieve_from_data_row(
#                     clean_annotation)
#
#                 simple_config = SimpleConfiguration(scene, figure, ground)
#
#                 for c in config_list:
#
#                     if simple_config.configuration_match(c):
#                         if clean_user_id not in user_dict[c]:
#                             user_dict[c].append(clean_user_id)
#
#         return user_dict
#
#     # Add users to config ratios
#     def write_config_ratios(self):
#         """Summary
#
#         ## Write csvs for each preposition giving feature values and data on number of selections
#         """
#         # first get users tested on each configuration
#
#         user_dict = self.get_tested_users_dict()
#         config_list = self.study_info.config_list
#
#         for preposition in self.get_used_prepositions():
#             new_csv = TEST_FOLDER + '/' + self.study_info.config_ratio_csv(self.filetag, preposition)
#             ## Write file of all instances
#             with open(new_csv, "w") as csvfile:
#                 outputwriter = csv.writer(csvfile)
#                 outputwriter.writerow(['Scene', 'Figure', 'Ground'] + self.feature_keys + [self.ratio_feature_name,
#                                                                                            self.categorisation_feature_name,
#                                                                                            'users'])
#
#                 for c in config_list:
#                     row = c.full_row.copy()
#
#                     t = float(c.number_of_tests(self.config_test_list))
#                     s = float(c.number_of_selections_from_instancelist(preposition, self.instance_list))
#
#                     ## If at least one test has been done for this configuration
#                     if t != 0:
#
#                         ratio = s / t
#
#                         r = str(ratio)
#
#                         row.append(r)
#
#                         if (ratio == 0):
#                             row.append(str(0))
#                         else:
#                             row.append(str(1))
#
#                         # Get users for the configuration
#                         users = user_dict[c]
#
#                         users_string = ';'.join(users)
#
#                         row.append(users_string)
#
#                         outputwriter.writerow(row)


# First models need redefining to use users rather than scenes

class NewGeneratePrepositionModelParameters(GeneratePrepositionModelParameters):

    def __init__(self, study_info_, preposition, train_scenes, features_to_remove=None, polyseme=None):
        new_csv = new_config_ratio_csv(preposition, study_info_)
        new_dataset = pd.read_csv(new_csv)
        GeneratePrepositionModelParameters.__init__(self, study_info_, preposition, train_scenes,
                                                    features_to_remove=features_to_remove, polyseme=polyseme,
                                                    given_dataset=new_dataset)

    def remove_nontrainingscenes(self, d):

        return d


# create modelgenerator where test_scenes are all scenes.
class GenerateModelsSplitUsers:

    def __init__(self, train_users, test_users, study_info):
        print(train_users)
        print(len(train_users))
        print(test_users)
        print(len(test_users))
        # TO cover the case where full scene list is passed
        if train_users == study_info.scene_name_list:
            train_users = study_info.user_name_list
        if test_users == study_info.scene_name_list:
            test_users = study_info.user_name_list

        self.test_users = test_users
        self.train_users = train_users
        self.study_info = study_info
        self.create_new_train_dataset(train_users)
        self.new_constraint_csv = new_constraint_csv(study_info)
        self.create_new_test_dataset(test_users)

    def create_new_train_dataset(self, train_users):
        print('writing new train set')
        print(train_users)
        print(len(train_users))
        # Need to call this every train/test split
        newsvcollection = SemanticCollection(self.study_info)

        filename = newsvcollection.study_info.data_folder + "/" + newsvcollection.study_info.sem_annotations_name

        # Create new lists from clean csv
        with open(filename, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            annotationlist = list(reader)  # create a list from the reader

            annotationlist.pop(0)  # removes first line of data list which is headings

            for clean_annotation in annotationlist:

                an_id, clean_user_id, task, scene, prepositions, figure, ground, time = SemanticAnnotation.retrieve_from_data_row(
                    clean_annotation)

                if clean_user_id in train_users:
                    simple_config = SimpleConfiguration(scene, figure, ground)
                    newsvcollection.config_test_list.append(simple_config)

                    for p in prepositions:
                        if p != "":
                            i = Instance(an_id, clean_user_id, task, scene, p, figure, ground,
                                         newsvcollection.study_info)

                            newsvcollection.instance_list.append(i)
        # newsvcollection.append_values()

        # Write new raio csv
        config_list = newsvcollection.study_info.config_list

        for preposition in PREPOSITION_LIST:
            new_csv = new_config_ratio_csv(preposition, newsvcollection.study_info)
            ## Write file of all instances
            with open(new_csv,
                      "w") as csvfile:
                outputwriter = csv.writer(csvfile)
                outputwriter.writerow(
                    ['Scene', 'Figure', 'Ground'] + newsvcollection.feature_keys + [newsvcollection.ratio_feature_name,
                                                                                    newsvcollection.categorisation_feature_name])

                for c in config_list:
                    row = c.full_row.copy()

                    t = float(c.number_of_tests(newsvcollection.config_test_list))
                    s = float(c.number_of_selections_from_instancelist(preposition, newsvcollection.instance_list))

                    ## If at least one test has been done for this configuration
                    if t != 0:

                        ratio = s / t

                        r = str(ratio)

                        row.append(r)

                        if (ratio == 0):
                            row.append(str(0))
                        else:
                            row.append(str(1))

                        outputwriter.writerow(row)

    def create_new_test_dataset(self, test_users):
        print('writing new test set')
        print(test_users)
        print(len(test_users))
        newcompcollection = ComparativeCollection(self.study_info)

        ### Reads annotations from clean files
        filename = self.study_info.data_folder + "/" + self.study_info.comp_annotations_name

        with open(filename, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            annotationlist = list(reader)  # create a list from the reader

        annotationlist.pop(0)  # removes first line of data list which is headings

        for clean_annotation in annotationlist:
            an_id, clean_user_id, task, scene, preposition, figure, ground, time, possible_figures = ComparativeAnnotation.retrieve_from_data_row(
                clean_annotation)

            if clean_user_id in test_users:
                i = CompInstance(an_id, clean_user_id, task, scene, preposition, figure, ground, possible_figures,
                                 self.study_info)

                newcompcollection.instance_list.append(i)
        newcompcollection.append_values()

        # Now write constraints
        ## Creates a dictionary, prepositions are keys
        ### Values are lists of constraints for the preposition
        out = dict()

        for preposition in PREPOSITION_LIST:
            # print(preposition)
            preposition_constraints = []
            for my_scene in self.study_info.scene_list:

                grounds = my_scene.selectable_objects

                for grd in grounds:

                    c = Comparison(my_scene.name, preposition, grd, self.study_info)
                    Cons = c.generate_constraints(newcompcollection.instance_list)
                    for con in Cons:
                        preposition_constraints.append(con)
            out[preposition] = preposition_constraints

            for con in preposition_constraints:
                con.write_to_csv(self.new_constraint_csv)

        return out


class GenerateBasicModelsSplitUsers(GenerateModelsSplitUsers):
    """Summary


    """


    # Generating models to test
    def __init__(self, train_users, test_users, study_info_, extra_features_to_remove=None, only_test_our_model=None,
                 test_prepositions=PREPOSITION_LIST):
        """Summary


        """
        GenerateModelsSplitUsers.__init__(self, train_users, test_users, study_info_)

        # Test and train on all scenes
        self.train_scenes = self.study_info.scene_name_list
        self.test_scenes = self.study_info.scene_name_list

        # Make sure train and test are distinct, if not using all users
        if self.train_users != self.study_info.user_name_list:
            f1_set = set(self.train_users)
            f2_set = set(self.test_users)
            if f1_set & f2_set:
                raise ValueError('Train and test scenes not distinct')

        # Features to remove from consideration (not used in training or testing)
        self.features_to_remove = Configuration.object_specific_features.copy()

        # Extra features may be removed in order to compare performance
        if extra_features_to_remove is not None:
            for f in extra_features_to_remove:
                self.features_to_remove.append(f)

        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in test_prepositions:
            M = NewGeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
                                                      features_to_remove=self.features_to_remove)
            M.work_out_models()
            preposition_models_dict[p] = M

        self.preposition_parameters_dict = preposition_models_dict

        # When models are generated, pass constraint csv for them to read.
        new_cons_csv = new_constraint_csv(self.study_info)
        our_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info,
                                   constraint_csv_removed_users=new_cons_csv)

        if only_test_our_model is None:

            exemplar_model = ExemplarModel(preposition_models_dict, self.test_scenes, self.study_info,
                                   constraint_csv_removed_users=new_cons_csv)
            cs_model = CSModel(preposition_models_dict, self.test_scenes, self.study_info,
                                   constraint_csv_removed_users=new_cons_csv)
            proximity_model = ProximityModel(self.test_scenes, self.study_info,
                                   constraint_csv_removed_users=new_cons_csv)
            simple_model = SimpleModel(self.test_scenes, self.study_info,
                                   constraint_csv_removed_users=new_cons_csv)
            best_guess_model = BestGuessModel(self.test_scenes, self.study_info,
                                   constraint_csv_removed_users=new_cons_csv)

            models = [our_model, exemplar_model, cs_model, proximity_model, simple_model, best_guess_model]

        else:

            models = [our_model]

        self.models = models
        self.model_name_list = []
        for m in self.models:
            self.model_name_list.append(m.name)


class MultipleRunsSplitParticipants(MultipleRuns):
    def __init__(self, model_generator, scores_tables_folder, scores_plots_folder, study_info_,
                 test_prepositions=PREPOSITION_LIST, number_runs=None,
                 k=None, compare=None):
        MultipleRuns.__init__(self, model_generator, scores_tables_folder, scores_plots_folder, study_info_,
                              test_prepositions=test_prepositions, number_runs=number_runs,
                              k=k, compare=compare)

    def folds_check(self, folds):
        """Summary

        Args:
            folds (TYPE): Description

        Returns:
            TYPE: Description
        """

        return True

    def get_validation_scene_split(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Get train-test users

        if self.k is not None:
            # Create random folds for testing
            folds = []

            users_left = self.study_info.user_name_list
            divisor = self.k
            while divisor > 1:
                t_size = float(1) / divisor
                train_users, test_users = train_test_split(users_left, test_size=t_size)
                folds.append(test_users)
                users_left = train_users
                divisor = divisor - 1
                if divisor == 1:
                    folds.append(train_users)

            if len(folds) == 0:
                raise Exception("No folds generated")

            return folds


def test_basic_models():
    study_info = StudyInfo("2019 study")
    all_scenes = study_info.scene_name_list
    # all_users = study_info.user_name_list
    # p_models = GenerateBasicModelsSplitUsers(all_users, all_users, study_info)
    #
    # models = p_models.models
    #
    # t = TestModels(models, "all")
    # all_dataframe = t.score_dataframe.copy()
    #
    # all_dataframe.to_csv(TEST_FOLDER + "/scores/all_test.csv")
    # print(all_dataframe)

    runs =100
    k=2
    m = MultipleRunsSplitParticipants(GenerateBasicModelsSplitUsers, TEST_FOLDER + "/scores",
                                      TEST_FOLDER + "/scores",
                                      study_info, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print(m.average_dataframe)

if __name__ == '__main__':
    test_basic_models()

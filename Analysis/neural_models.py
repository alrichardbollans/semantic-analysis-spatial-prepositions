import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from typing import Dict

from Analysis.data_import import StudyInfo, Configuration
from Analysis.performance_test_functions import MultipleRuns, ModelGenerator, TestModels, Model, compare_models
from baseline_model_testing import GeneratePrepositionModelParameters, GenerateBasicModels, PREPOSITION_LIST, \
    PrototypeModel, get_standard_preposition_parameters

NEURAL_MODEL_SCORES_FOLDER = "model evaluation/neural models"
NEURAL_MODEL_INFO_FOLDER = "model info/neural models"


class PerfectAccCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        # onepochend gets called whenever epoch ends
        if (logs.get('accuracy') == 1):
            # logs contains info about training state
            print("\nReached 100% accuracy so cancelling training!")
            self.model.stop_training = True


class HighAccCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        # onepochend gets called whenever epoch ends
        if (logs.get(NeuralNetworkCategorisationModel.performance_metric) < 0.01):
            # logs contains info about training state
            print("\nReached 0.01 error so cancelling training!")
            self.model.stop_training = True


perfect_acc_callback = PerfectAccCallback()
high_acc_callback = HighAccCallback()

val_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


class NeuralNetworkCategorisationModel(Model):
    # Want this model to train on SV data in a reliable way, and then use these categorisation predictions to make
    # typicality judgements.

    name = "Neural Net Classification"
    performance_metric = 'mean_squared_error'

    def __init__(self, preposition_model_dict: Dict[str, GeneratePrepositionModelParameters], test_scenes, study_info_,
                 train_test_proportion=float(1), number_of_epochs=200, make_plots=False, test_prepositions=None):
        Model.__init__(self, self.name, test_scenes, study_info_, test_prepositions=test_prepositions)
        self.models = dict()
        self.preposition_model_dict = preposition_model_dict
        self.training_data_dict = {}
        self.callbacks = None
        self.number_of_epochs = number_of_epochs
        self.train_test_proportion = train_test_proportion
        self.make_plots = make_plots

        for p in self.test_prepositions:

            self.training_data_dict[p] = self.convert_train_dataframe_to_tfdataset(
                preposition_model_dict[p].train_dataset, p)

            if self.train_test_proportion == 1:
                train = self.training_data_dict[p]
                model = self.train_model(p, train)
                self.callbacks = [high_acc_callback]
            else:
                train_amount = int(len(self.training_data_dict[p]) * self.train_test_proportion)
                train = self.training_data_dict[p].take(train_amount)
                test = self.training_data_dict[p].skip(train_amount)
                model = self.train_model(p, train, test)
            self.models[p] = model

    def convert_train_dataframe_to_tfdataset(self, df, preposition):

        # The data is first oversampled to improve categorisation of (rare) positive instances.
        copy_df = df.copy()
        positive_examples = copy_df[(copy_df.iloc[:, GeneratePrepositionModelParameters.category_index] == 1)]
        oversampled_df = pd.concat([copy_df, positive_examples], ignore_index=True)
        oversampled_df = pd.concat([oversampled_df, positive_examples], ignore_index=True)

        removed_nonfeatures = oversampled_df.drop(
            ["Scene", "Figure", "Ground",
             GeneratePrepositionModelParameters.categorisation_feature_name], axis=1)
        remove_unused_features: pd.DataFrame = self.preposition_model_dict[preposition].remove_unused_features(
            removed_nonfeatures)

        target = remove_unused_features.pop(GeneratePrepositionModelParameters.ratio_feature_name)

        training_dataset_unbatched = tf.data.Dataset.from_tensor_slices((remove_unused_features.values, target.values))

        train_dataset = training_dataset_unbatched.shuffle(len(remove_unused_features)).batch(1)

        return train_dataset

    def train_model(self, preposition, train_data, val_data=None):
        # THe model has a 10 dimensional input space, to account for each feature.
        # The output is a single neuron providing probability of the given preposition.
        # The model has a single hidden dense layer of 6 neurons
        # Hidden layers has relu activation.
        # Mean squared error loss is used ( to train the model) and the model is stopped when high training accuracy is achieved
        # We will see that the model fits the data and generalises well.

        number_of_features = len(self.preposition_model_dict[preposition].feature_keys)

        model = tf.keras.models.Sequential([
            # Relu effectively means "If X>0 return X, else return 0" --
            # so what it does it it only passes values 0 or greater to the next layer in the network.
            keras.layers.InputLayer(input_shape=(number_of_features,)),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=[self.performance_metric])

        print(f'fitting:{preposition}')
        history = model.fit(train_data, validation_data=val_data, epochs=self.number_of_epochs,
                            callbacks=self.callbacks, verbose=0)

        acc = history.history[self.performance_metric]

        loss = history.history['loss']
        epochs = range(len(acc))
        if self.make_plots:
            if val_data is not None:
                val_loss = history.history['val_loss']
                plt.plot(epochs, val_loss, 'g', label='Validation Loss')

            plt.plot(epochs, loss, 'm', label='Training Loss')
            # plt.ylim((0, 1))
            plt.title('Training and validation loss: ' + preposition)
            plt.legend(loc=0)

            plt.savefig(f'{NEURAL_MODEL_INFO_FOLDER}/plots/{self.name}/{preposition}.pdf')
            plt.show()
            model.summary()

        # for element in train_data:
        #     if element[1] > 0:
        #         print(element[1])
        #         print(model.predict(element[0]))

        return model

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):

        value_array = list(value_array)
        new_array = self.remove_features_from_array(value_array,
                                                    self.preposition_model_dict[preposition].features_to_remove)

        # print(new_array)
        new_array = tf.convert_to_tensor([new_array])

        # print(new_array)
        return self.models[preposition].predict(new_array)


class SupervisedNeuralTypicalityModel(Model):
    name = "Supervised Model"

    # This model will take constraint dict as input and train to guess correct configuration.

    def __init__(self, train_scenes, test_scenes, study_info_, features_to_remove, train_test_proportion=float(1),
                 number_of_epochs=200, make_plots=False, test_prepositions=None):
        Model.__init__(self, self.name, test_scenes, study_info_, test_prepositions=test_prepositions)
        self.models = dict()
        self.train_scenes = train_scenes
        self.features_to_remove = features_to_remove
        self.number_of_epochs = number_of_epochs
        self.train_test_proportion = train_test_proportion
        self.make_plots = make_plots

        self.models = dict()
        self.train_datasets = dict()
        self.callbacks = None

        for p in self.test_prepositions:
            self.train_datasets[p] = self.prepare_train_dataset(p)
            if self.train_test_proportion == 1:
                train = self.train_datasets[p]
                model = self.train_model(p, train)
                self.callbacks = [perfect_acc_callback]
            else:
                train_amount = int(len(self.train_datasets[p]) * self.train_test_proportion)
                train = self.train_datasets[p].take(train_amount)
                test = self.train_datasets[p].skip(train_amount)
                model = self.train_model(p, train, test)

            self.models[p] = model

    def get_train_constraints(self, preposition):
        all_constraints = self.constraint_dict[preposition]
        # Constraints to train on
        train_constraints = []

        for c in all_constraints:
            if c.scene in self.train_scenes:
                train_constraints.append(c)
        return train_constraints

    def prepare_train_dataset(self, preposition):
        training_arrays = []
        targets = []

        train_constraints = self.get_train_constraints(preposition)
        for c in train_constraints:

            train_array = np.subtract(c.lhs_values, c.rhs_values)
            train_array = self.remove_features_from_array(train_array, self.features_to_remove)

            for i in range(c.weight):
                # oversample to account for weight
                training_arrays.append(train_array)
                targets.append(1)

        # Now invert half of the training examples as they
        # are currently all positive, i.e. the model could just always guess 1
        indices = range(len(training_arrays))
        random_indices = random.sample(indices, int(len(indices) / 2))

        for i in random_indices:
            training_arrays[i] = [-x for x in training_arrays[i]]
            targets[i] = 0

        training_dataset_unbatched = tf.data.Dataset.from_tensor_slices((training_arrays, targets))

        train_dataset = training_dataset_unbatched.shuffle(len(training_arrays)).batch(1)

        return train_dataset

    def train_model(self, preposition, train_data, val_data=None):
        # print(list(train_data.as_numpy_iterator())[0][0].shape)
        number_of_features = len(self.all_feature_keys) - len(Configuration.object_specific_features)

        if number_of_features != 10:
            raise ValueError("Incorrect number of features.")

        model = tf.keras.models.Sequential([
            # Relu effectively means "If X>0 return X, else return 0" --
            # so what it does it it only passes values 0 or greater to the next layer in the network.
            keras.layers.InputLayer(input_shape=(number_of_features,)),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['accuracy'])

        history = model.fit(train_data, validation_data=val_data, epochs=self.number_of_epochs,
                            callbacks=self.callbacks, verbose=0)

        acc = history.history['accuracy']

        loss = history.history['loss']
        epochs = range(len(acc))

        if self.make_plots:
            if val_data is not None:
                val_acc = history.history['val_accuracy']
                val_loss = history.history['val_loss']
                plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
                plt.plot(epochs, val_loss, 'g', label='Validation Loss')

            plt.plot(epochs, acc, 'r', label='Training accuracy')

            plt.plot(epochs, loss, 'm', label='Training Loss')
            plt.ylim((0, 1))
            plt.title('Training and validation accuracy: ' + preposition)
            plt.legend(loc=0)
            plt.figure()

            plt.show()

            model.summary()

        return model

    def weighted_score(self, Constraints):
        """Summary

        Args:
            preposition (TYPE): Description
            Constraints (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Calculates how well W and P satisfy the constraints, accounting for constraint weight
        counter = 0

        for c in Constraints:
            test_array = np.subtract(c.lhs_values, c.rhs_values)

            if self.get_typicality(c.preposition, test_array) > 0.5:
                counter += c.weight

        return counter

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):

        value_array = list(value_array)
        new_array = self.remove_features_from_array(value_array,
                                                    self.features_to_remove)

        # print(new_array)
        new_array = tf.convert_to_tensor([new_array])

        # print(new_array)
        return self.models[preposition].predict(new_array)


class GenerateNeuralModels(GenerateBasicModels):

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions):
        GenerateBasicModels.__init__(self, train_scenes, test_scenes, study_info_, test_prepositions=test_prepositions)

        preposition_models_dict = self.preposition_parameters_dict

        self.neural_categorisation = NeuralNetworkCategorisationModel(preposition_models_dict, self.test_scenes,
                                                                      self.study_info,
                                                                      test_prepositions=test_prepositions)

        # self.neural_supervised = SupervisedNeuralTypicalityModel(self.train_scenes, self.test_scenes, self.study_info,
        #                                                          self.features_to_remove)

        # self.baseline_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info)

        self.generate_model_lists()


def verify_dnn_model():
    # Show the model trains well
    study_info = StudyInfo("2019 study")
    scene_list = study_info.scene_name_list

    preposition_models_dict = get_standard_preposition_parameters(scene_list)

    dnn_model = NeuralNetworkCategorisationModel(preposition_models_dict, scene_list, study_info, make_plots=True,
                                                 train_test_proportion=0.8, number_of_epochs=1000)

    t = TestModels([dnn_model], "all")
    all_dataframe = t.score_dataframe.copy()

    print(all_dataframe)


def verify_sup_model():
    # Show the model trains well
    study_info = StudyInfo("2019 study")
    scene_list = study_info.scene_name_list
    features_to_remove = Configuration.object_specific_features.copy()
    train_scenes, test_scenes = scene_list, scene_list
    model = SupervisedNeuralTypicalityModel(train_scenes, test_scenes, study_info, features_to_remove,
                                            train_test_proportion=0.8, number_of_epochs=1000)

    t = TestModels([model], "all")
    all_dataframe = t.score_dataframe.copy()

    print(all_dataframe)


def test_neural_models(runs, k):
    """Summary

    Args:
        runs (TYPE): Description
        k (TYPE): Description
        study_info_ (TYPE): Description
        :param study_info_:
    """
    compare_models(runs, k, GenerateNeuralModels, NEURAL_MODEL_SCORES_FOLDER)


def main():
    print(tf.__version__)
    # should_be_one = [1.33, -0.60, 1.99, 0.40, -0.75, 2.60, -0.95, 0.45, 1.51, 1.84]
    # print(tf.shape(should_be_one))
    # print(tf.shape([should_be_one]))

    # sup_dnn_model_training()
    # verify_dnn_model()
    # verify_sup_model()

    #
    test_neural_models(10, 10)


if __name__ == '__main__':
    main()

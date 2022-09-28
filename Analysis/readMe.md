# Info

Scripts given here are used to process and analyse the data.

Versions used:
Python 3
Pandas 1.0.3

Explanation of some folders:

## Folders

The 2019 study folder contains the following:

* Scene Data: Contains information and features extracted from the study scenes
* collected data: Collection of raw and cleaned annotation lists.
* constraint data: List of constraints generated from data. This is created using compile_instances.py and read in basic_model_testing.py
* feature values: Contains information on feature values from configurations in all scenes. Standardised values are created by
preprocess_features.py and read in compile_instances.py
* preposition data: Data related to specific prepositions. Giving configurations and how often they are selected with a given preposition in
sv_task
* stats: Collection of general stats giving overview of each task

The model info folder contains a collection of basic model parameters when trained on all scenes and categorisation accuracy of the perceptron models.

The model evaluation folder contains results of cross validation tests


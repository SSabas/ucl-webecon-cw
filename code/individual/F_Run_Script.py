"""
Project:
    COMPGW02/M041 Web Economics Coursework Project

Description:
    In this assignment, we are required to work on an online advertising problem. We will help advertisers to form
    a bidding strategy in order to place their ads online in a realtime bidding system. We are required to train a
    bidding strategy based on a provided advertising impression training set. This project aims to help us understand
    some basic concepts and write a computer program in real-time bidding based display advertising. As we will be
    evaluated both as a group as well as individually, part of the assignment is to train a model of our choice
    independently. The performance of the model trained by the team, which is either a combination of the
    individually developed models or the best performing individually-developed model, will be (mainly) evaluated
    on the Click-through Rate achieved on a provided test set.

Authors:
  Sven Sabas

Date:
  22/02/2018
"""

# ------------------------------ IMPORT LIBRARIES --------------------------------- #

import pandas as pd
import os
import sys
import numpy as np

# ------------------------ ADD CWD TO PYTHONPATH ---------------------------------- #

# For module importing
working_dir = os.getcwd() + ('/code/individual')
sys.path.append(working_dir)

# Set display settings
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)

# --------------------------------- GET DATA -------------------------------------- #

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
validation = pd.read_csv('./data/validation.csv')

# ---------------------------- EXPLORATORY ANALYSIS ------------------------------- #


# ---------------------------- FEATURE ENGINEERING -------------------------------- #

# Get functions from data preprocessing script
from B_Data_Preprocessing import *

# NB! DO FEATURE ENGINEERING TOGETHER FOR ALL 3 DATASETS (TRAINING, VALIDATION AND
# TEST) TO GET COHERENT REPRESENTATION!

# Merge the datasets
data = merge_datasets(train, validation, test)

# data the data preprocessing function
data = add_features(data)
data = exclude_irrelevant_features(data)
data, label_dictionary = label_encoder(data)
data = one_hot_encoding(data)
data = min_max_scaling(data)

# Separate the datasets
train1, validation1, test1 = separate_datasets(data, train, validation, test)

# Upsample the minority class
train2 = downsampling_majority_class(train1, class_ratio=0.05, seed=500)


# ---------------------------- CTR PREDICTION ------------------------------------------ #

# --- SOME TOGGLES FOR ANALYSIS

run_gridsearch = 'no'
import_model_parameters = 'no'

# --- LOGISTIC MODEL --- #
log_classifier, log_prediction = logistic_model(train2, validation1, use_gridsearch='no', refit='yes', refit_iter=500,
                                                use_saved_model='yes', save_model='no', to_plot='yes', random_seed=500,
                                                parameters={'C': [0.1, 0.5, 1, 2, 5, 10], 'penalty': ['l1', 'l2'],
                                                            'class_weight': ['unbalanced', 'balanced'], 'tol': [0.0001],
                                                            'solver': ['saga'], 'max_iter': [100]})
# --- RANDOM FOREST --- #
rf_classifier, rf_prediction = random_forest(train2, validation1, use_gridsearch='no', refit='yes', refit_iter=200,
                                             use_saved_model='yes', save_model='no', to_plot='yes', random_seed=500,
                                             parameters={'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None],
                                                         'min_samples_split' :[4,5,6],
                                                         "n_estimators" : [50],
                                                         "min_samples_leaf": [1,2,3,4,5],
                                                         "max_features": [4,5,6,"sqrt"],
                                                         "criterion": ['gini'],
                                                         'random_state': [500]})

# --- EXTREME RANDOM FOREST --- #
erf_classifier, erf_prediction = extreme_random_forest(train2, validation1, use_gridsearch='no', refit='yes',
                                                       refit_iter=200, use_saved_model='yes', save_model='no',
                                                       to_plot='yes', random_seed=500,
                                                       parameters={'max_depth': [2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None],
                                                                   'min_samples_split': [4, 5, 6],
                                                                   "n_estimators": [50],
                                                                   "min_samples_leaf": [1, 2, 3, 4, 5],
                                                                   "max_features": [4, 5, 6, "sqrt"],
                                                                   "criterion": ['gini']})

# --- XGBOOST --- #
xgb_classifier, xgb_prediction = gradient_boosted_trees(train2, validation1, use_gridsearch='no', refit='yes',
                                                        refit_iter=400, use_saved_model='yes', save_model='yes',
                                                        to_plot='yes', random_seed=500,
                                                        parameters={'max_depth': [15, 20, 25, 30], "n_estimators": [20],
                                                                    "learning_rate": [0.05, 0.1, 0.2],
                                                                    "colsample_bytree": [0.5, 0.8, 1],
                                                                    "reg_alpha": [0.1], "reg_lambda": [0.1],
                                                                    "subsample": [0.5, 0.8, 1], "gamma": [0, 0.1]})
# --- SUPPORT VECTOR MACHINES --- #
svm_classifier, svm_prediction = support_vector_machine(train2, validation1, use_gridsearch='no', refit='yes',
                                                        refit_iter=50, use_saved_model='yes', save_model='yes',
                                                        to_plot='yes', random_seed=500,
                                                        parameters={'C': [0.1, 1, 2],
                                                                    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                                                                    "degree": [2, 3, 4],
                                                                    "gamma": ['auto'],
                                                                    "tol": [0.001],
                                                                    "max_iter": [10],
                                                                    "probability": [True],
                                                                    "cache_size": [1000]})

# --- NAIVE BAYES --- #
nb_classifier, nb_prediction = naive_bayes(train2, validation1, use_saved_model='no', save_model='yes', to_plot ='yes')

# --- FACTORIZATION MACHINES --- #
fm_classifier, fm_prediction = factorization_machine(train2, validation1, refit='yes',
                                                     refit_iter=20, use_saved_model='no', save_model='yes',
                                                     to_plot='yes', random_seed=500,
                                                     parameters={'init_stdev': 0.1, "rank": 2,
                                                                 'l2_reg_w': 0.1, 'l2_reg_V': 0.1,
                                                                 'n_iter': 300})

# --- STACKING MODEL --- #
stacked_classifier, stacked_prediction = stacking_classifier(train2, validation1, refit='yes', use_saved_model='yes',
                                                             save_model='no', to_plot='yes',
                                                             meta_leaner_parameters={'max_depth': 20, "n_estimators": 20,
                                                                                     "learning_rate": 0.05,
                                                                                     'silent': False, 'n_jobs': 3,
                                                                                     'subsample': 1,
                                                                                     'objective': 'binary:logistic',
                                                                                     'colsample_bytree': 1,
                                                                                     'eval_metric': "auc",
                                                                                     'reg_alpha': 0.1,
                                                                                     'reg_lambda': 0.1,
                                                                                     'random_state': 500},
                                                             stacking_cv_parameters={'use_probas': False,
                                                                                     'use_features_in_secondary': True,
                                                                                     'cv': 2,
                                                                                     'store_train_meta_features': True,
                                                                                     'refit': True})

# --- COMPARE THE AUC  (PLOT ROC CURVES ON SAME GRAPH) --- #

# ---------------------------- BIDDING STRATEGY ---------------------------------------- #

# ---------------------------- OUTPUT  ------------------------------------------------- #

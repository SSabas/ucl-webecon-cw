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
import curl

# ------------------------ ADD CWD TO PYTHONPATH ---------------------------------- #

# For module importing
working_dir = os.getcwd() + ('/code')
sys.path.append(working_dir)

# Set display settings
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)

# --- SOME TOGGLES FOR ANALYSIS

run_gridsearch = 'yes' #'no'
use_saved_model = 'no' #'no'
save_model = 'yes'
minority_class = 0.2

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
train2 = downsampling_majority_class(train1, class_ratio=minority_class, seed=500)

# ---------------------------- CTR PREDICTION ------------------------------------------ #

# Get functions from CTR prediction script
from C_CTR_Prediction import *

# --- LOGISTIC MODEL --- #
log_classifier, log_prediction = logistic_model(train2, validation1, use_gridsearch=run_gridsearch, refit='yes', refit_iter=500,
                                                use_saved_model=use_saved_model, save_model=save_model, to_plot='yes', random_seed=500,
                                                parameters={'C': [0.1, 0.2, 0.5], 'penalty': ['l1', 'l2'],
                                                            'class_weight': ['unbalanced'], 'tol': [0.001],
                                                            'solver': ['saga'], 'max_iter': [100]})
# --- RANDOM FOREST --- #
rf_classifier, rf_prediction = random_forest(train2, validation1, use_gridsearch=run_gridsearch, refit='yes', refit_iter=1000,
                                             use_saved_model=use_saved_model, save_model=save_model, to_plot='yes', random_seed=500,
                                             parameters={'max_depth': [5, 10, 11, 12, None],
                                                         'min_samples_split' :[4, 6, 8],
                                                         "n_estimators" : [200],
                                                         "min_samples_leaf": [1, 3, 5],
                                                         "max_features": [5, 20, "sqrt"],
                                                         "criterion": ['gini'],
                                                         'random_state': [500]})

# --- EXTREME RANDOM FOREST --- #
erf_classifier, erf_prediction = extreme_random_forest(train2, validation1, use_gridsearch=run_gridsearch, refit='yes',
                                                       refit_iter=1000, use_saved_model=use_saved_model, save_model=save_model,
                                                       to_plot='yes', random_seed=500,
                                                       parameters={'max_depth': [10, 20, None],
                                                                   'min_samples_split': [2, 5, 10],
                                                                   "n_estimators": [200],
                                                                   "min_samples_leaf": [2, 5, 10],
                                                                   "max_features": [5, 20, "sqrt"],
                                                                   "criterion": ['gini']})

# --- XGBOOST --- #
xgb_classifier, xgb_prediction = gradient_boosted_trees(train2, validation1, use_gridsearch=run_gridsearch, refit='yes',
                                                        refit_iter=500, use_saved_model=use_saved_model, save_model=save_model,
                                                        to_plot='yes', random_seed=500,
                                                        parameters={'max_depth': [20], "n_estimators": [50],
                                                                    "learning_rate": [0.1],
                                                                    "colsample_bytree": [0.8],
                                                                    "reg_alpha": [0.1], "reg_lambda": [0.1],
                                                                    "subsample": [1], "gamma": [0]})
# --- SUPPORT VECTOR MACHINES --- #
svm_classifier, svm_prediction = support_vector_machine(train2, validation1, use_gridsearch=run_gridsearch, refit='yes',
                                                        refit_iter=100, use_saved_model=use_saved_model, save_model=save_model,
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
                                                     refit_iter=500, use_saved_model='yes', save_model='no',
                                                     to_plot='yes', random_seed=500,
                                                     parameters={'init_stdev': 0.1, "rank": 2,
                                                                 'l2_reg_w': 0.1, 'l2_reg_V': 0.1,
                                                                 'n_iter': 300})

# --- STACKING MODEL --- #
stacked_classifier, stacked_prediction = stacking_classifier(train2, validation1, refit='yes', use_saved_model='no',
                                                             save_model='yes', to_plot='yes',
                                                             meta_leaner_parameters={'max_depth': 20, "n_estimators": 500,
                                                                                     "learning_rate": 0.05,
                                                                                     'silent': False, 'n_jobs': 3,
                                                                                     'subsample': 1,
                                                                                     'objective': 'binary:logistic',
                                                                                     'colsample_bytree': 0.8,
                                                                                     'eval_metric': "auc",
                                                                                     'reg_alpha': 0.1,
                                                                                     'reg_lambda': 0.1,
                                                                                     'random_state': 500},
                                                             stacking_cv_parameters={'use_probas': False,
                                                                                     'use_features_in_secondary': True,
                                                                                     'cv': 5,
                                                                                     'store_train_meta_features': True,
                                                                                     'refit': True})



# --- COMPARE THE AUC  (PLOT ROC CURVES ON SAME GRAPH) --- #
plot_ROC_curve(validation1['click'], log_prediction, model='Logistic', minority_class=minority_class)
plot_ROC_curve(validation1['click'], rf_prediction, model='Random Forest', minority_class=minority_class)
plot_ROC_curve(validation1['click'], erf_prediction, model='Extreme Random Forest', minority_class=minority_class)
plot_ROC_curve(validation1['click'], xgb_prediction, model='XGBoost', minority_class=minority_class)
plot_ROC_curve(validation1['click'], svm_prediction, model='SVM', minority_class=minority_class)
plot_ROC_curve(validation1['click'], nb_prediction, model='Naive Bayes', minority_class=minority_class)
plot_ROC_curve(validation1['click'], fm_prediction, model='Factorization Machine', minority_class=minority_class)
plot_ROC_curve(validation1['click'], stacked_prediction, model='Stacked', minority_class=minority_class)
plt.savefig(os.getcwd()+'/results/AUC_comparison_'+str(int(minority_class*100))+'.pdf')


# ---------------------------- TEST DOWNSAMPLING EFFECT ---------------------------------------- #

downsampling_sensitivity = test_downsampling(train1, validation1, stacked_classifier,
                                             minority_levels=np.linspace(0.01, 0.2, 5),
                                             model_type='Stacked', random_seed=500)
plt.savefig(os.getcwd()+'/results/downsizing_sensitivity.pdf')

# ---------------------------- BIDDING STRATEGY ---------------------------------------- #

# Get functions from Bidding Strategies script
from D_Bidding_Strategies import *

# Run the grid search for hyperparameters

# --- CONSTANT BIDDING --- #
constant_output = strategy_evaluation(validation1, erf_prediction, parameter_range=np.linspace(200, 400, 100),
                                      type='constant', budget=625000, to_plot='yes')

# --- RANDOM BIDDING --- #
a = np.tile(np.linspace(100, 299, 50), 50)
b = np.repeat(np.linspace(300, 700, 50), 50)
random_output = strategy_evaluation(validation1, stacked_prediction, parameter_range=np.column_stack((a, b)),
                                    type='random', budget=625000, to_plot='yes', plot_3d='yes', repeated_runs=20)

# --- LINEAR BIDDING --- #
linear_output = strategy_evaluation(validation1, stacked_prediction, parameter_range=np.linspace(200, 400, 1000),
                                    type='linear', budget=625000, to_plot='yes')

# --- SQUARE BIDDING --- #
square_output = strategy_evaluation(validation1, stacked_prediction, parameter_range=np.linspace(0.0001, 0.004, 1000),
                                    type='square', budget=625000, to_plot='yes')

# --- EXPONENTIAL BIDDING --- #
exponential_output = strategy_evaluation(validation1, stacked_prediction, parameter_range=np.linspace(0.0001, 0.2, 1000),
                                    type='exponential', budget=625000, to_plot='yes')

# --- ORTB1 BIDDING --- #
b = np.tile(np.linspace(5e-10, 3e-7, 100), 100)
a = np.repeat(np.linspace(0.001, 0.02, 100), 100)
ORTB1_output = strategy_evaluation(validation1, stacked_prediction, parameter_range=np.column_stack((a, b)),
                                    type='ORTB1', budget=625000, to_plot='yes', plot_3d='yes')


# --- ORTB2 BIDDING --- #
b = np.tile(np.linspace(5e-7, 6e-6, 100), 100)
a = np.repeat(np.linspace(1, 30, 100), 100)
ORTB2_output = strategy_evaluation(validation1, stacked_prediction, parameter_range=np.column_stack((a, b)),
                                    type='ORTB2', budget=625000, to_plot='yes', plot_3d='yes', repeated_runs=20)

# ---------------------------- OUTPUT  ------------------------------------------------- #

# Retrain the model using train plus validation data
train_plus_validation = pd.concat([train1, validation1])
train_plus_validation = train1.append(validation1)
train_plus_validation = downsampling_majority_class(train_plus_validation, class_ratio=minority_class, seed=500)

# Refit the model with new training data
refitted_model = stacked_classifier.fit(train_plus_validation.drop(['click', 'bidprice', 'payprice'], axis=1).values, train_plus_validation['click'].values)

# Predict for the testing set using best model (ERF in our case) plus train and validation data together
test_prediction = refitted_model.predict_proba(test1.drop(['click', 'bidprice', 'payprice'], axis=1).values)[:, 1]

# Use best bidding strategy (linear) to get the bids
# Get the coefficient and avgCTR for best linear model
linear_parameters = math.floor(linear_output.ix[linear_output['clicks_won'].idxmax()][2])
train_plus_validation_full = train.append(validation)
avgCTR = np.repeat(np.sum(train_plus_validation_full['click'] == 1) / train_plus_validation_full.shape[0], test_prediction.shape[0])

# Get bid prices
bids = np.repeat(linear_parameters*1.2, test_prediction.shape[0]) * (np.array(test_prediction) + avgCTR)

# Output results in csv file compatible with the submission
submission = pd.DataFrame(np.asarray([np.array(test.bidid), bids]).T, columns=['bidid', 'bidprice'])
submission.to_csv(os.getcwd()+"/results/testing_bidding_price.csv", index=False)

# Submit electronically
# curl http://deepmining.cs.ucl.ac.uk/api/upload/wining_criteria_1/92ZX62SoMlVG -X Post -F 'file=@/Users/ssabas/Desktop/ucl-webecon-cw/results/testing_bidding_price.csv'

####################### END ########################
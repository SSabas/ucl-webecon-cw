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
train1 = upsampling_minority_class(train1, class_ratio=0.02, seed=500)


# ---------------------------- CTR PREDICTION ------------------------------------------ #

model = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight = 'unbalanced', verbose = 2,
                           max_iter = 20, n_jobs = 1, tol=0.0025)

param_grid = {'C': [0.1, 1, 10],
              'penalty': ['l1','l2'],
              'class_weight': ['balanced', 'unbalanced']}

model = model.fit(train1.drop(['click','bidprice', 'payprice'], axis=1), train1['click'])


prediction = model.predict(validation1.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = model.predict_proba(validation1.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction_proba[:, 1])


plot_ROC_curve(validation['click'], prediction_proba[:, 1])


# Test how many iterations are needed
n_iter = 10
seed = 500
step_size = 1
values = np.arange(1, n_iter)

fm = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight = 'unbalanced', verbose = 10,
                           max_iter = 1, n_jobs = 3, tol=0.0025, warm_start = True)
# Initalize coefs
fm.fit(train1.drop(['click','bidprice', 'payprice'], axis=1), train1['click'])

roc_auc_train = []
roc_auc_validation = []
for i in range(1, n_iter):
    print(i)
    fm.fit(train1.drop(['click','bidprice', 'payprice'], axis=1), train1['click'])
    y_pred = fm.predict_proba(validation1.drop(['click', 'bidprice', 'payprice'], axis=1))
    roc_auc_validation.append(roc_auc_score(validation1['click'], y_pred))
    roc_auc_train.append(roc_auc_score(train1['click'], fm.predict_proba(train1.drop(['click','bidprice', 'payprice'], axis=1))))





x = np.arange(1, n_iter) * step_size

with plt.style.context('fivethirtyeight'):
    plt.plot(x, roc_auc_train, label='train')
    plt.plot(x, roc_auc_validation, label='test')
   # plt.plot(values, roc_auc_validation_re, label='train re', linestyle='--')
   # plt.plot(values, roc_auc_train_re, label='test re', ls='--')
plt.legend()
plt.show()



# ---------------------------- BIDDING STRATEGY ---------------------------------------- #


# ---------------------------- OUTPUT  ------------------------------------------------- #

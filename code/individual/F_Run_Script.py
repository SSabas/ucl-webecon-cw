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


# ---------------------------- MODELLING AND PREDICTION -------------------------------- #

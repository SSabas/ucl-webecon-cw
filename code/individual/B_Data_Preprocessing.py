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

#------------------------------ IMPORT LIBRARIES ---------------------------------#

import numpy as np
import pandas as pd
import re
import os
import csv
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#--------------------------------- GET DATA --------------------------------------#

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
validation = pd.read_csv('./data/validation.csv')

#-------------------- FUNCTIONS FOR FEATURE ENGINEERING --------------------------#

def separate_useragent(data):
    """
    Separates operating system and internet browser in useragent variable
    """

    # Split the useragent into two new columns
    data['opsys'], data['browser'] = data['useragent'].str.split('_').str

    # Remove column
    data = data.drop(['useragent'], axis=1)

    return data


def add_features(data):
    """
    Add new features to the dataset
    """

    # Separate useragent variable
    data = separate_useragent(data)

    # Calculate the area of an advert
    data['slotarea'] = data['slotwidth'] * data['slotheight']

    # Add count variables from usertag
    data['usertag'] = data['usertag'].str.split(',')
    data_usertags = data['usertag'].str.join('@').str.get_dummies('@').add_prefix('usertags_')
    data = pd.concat([data, data_usertags], axis=1)
    data = data.drop(['usertag'], axis=1)

    return data



def exclude_irrelevant_features(data, remove_columns = ['bidid', 'userid', 'IP',
                      'domain', 'url', 'urlid', 'slotid']):
    """
    Remove irrelevant columns with too specific values
    """

    data = data.drop(remove_columns, axis=1)

    return data


train = add_features(train)
train = exclude_irrelevant_features(train)


def label_encoder(data, columns_for_enconding = ['slotvisibility', 'slotformat', 'creative', 'keypage']):
    """
    Use binary features only
    """

    # Create dictionary of category mappings
    transform_dict = {}
    for col in columns_for_enconding:
        cats = pd.Categorical(data[col]).categories
        d = {}
        for i, cat in enumerate(cats):
            d[cat] = i
        transform_dict[col] = d

    # Create an inverse dictionary
    inverse_transform_dict = {}
    for col, d in transform_dict.items():
        inverse_transform_dict[col] = {v: k for k, v in d.items()}

    # Replace the odd values in initial dataset
    data[columns_for_enconding] = data[columns_for_enconding].replace(transform_dict)

    return data, inverse_transform_dict


train, label_dictionary = label_encoder(train)


def one_hot_encoding(data, columns_to_encode = ['weekday', 'hour', 'region',
                                                'city', 'adexchange', 'slotvisibility',
                                                'slotformat', 'creative', 'keypage',
                                                'advertiser', 'opsys', 'browser']):
    """
    One-hot encode multi-class columns
    """

    data = pd.get_dummies(data, columns=columns_to_encode)

    return data


train = one_hot_encoding(train)


def min_max_scaling(data, scale_columns = ['slotwidth', 'slotheight', 'slotprice',
                                           'slotarea']):
    """
    Scale the columns for better inference/training
    """
    mm = preprocessing.MinMaxScaler()
    data[scale_columns] = pd.DataFrame(mm.fit_transform(data[scale_columns]))

    return data



############################## END ##################################
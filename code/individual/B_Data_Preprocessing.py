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
from sklearn import preprocessing
from sklearn.utils import resample
import math

# -------------------- FUNCTIONS FOR FEATURE ENGINEERING -------------------------- #


def merge_datasets(train, validation, test):
    """
    Merges datasets for preprocessing
    """
    # Merge the datasets
    data = [train, validation, test]
    data = pd.concat(data)

    return data


def separate_datasets(merged, train, validation, test):
    """
    Separates preprocessed datasets
    """

    train = merged[:train.shape[0]]
    validation = merged[train.shape[0]: train.shape[0]+validation.shape[0]]
    test = merged[:-test.shape[0]]

    return train, validation, test


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


def one_hot_encoding(data, columns_to_encode = ['weekday', 'hour', 'region',
                                                'city', 'adexchange', 'slotvisibility',
                                                'slotformat', 'creative', 'keypage',
                                                'advertiser', 'opsys', 'browser']):
    """
    One-hot encode multi-class columns
    """

    data = pd.get_dummies(data, columns=columns_to_encode)

    return data


def min_max_scaling(data, scale_columns = ['slotwidth', 'slotheight', 'slotprice',
                                           'slotarea']):
    """
    Scale the columns for better inference/training
    """
    mm = preprocessing.MinMaxScaler()
    data[scale_columns] = pd.DataFrame(mm.fit_transform(data[scale_columns]))

    return data


def upsampling_minority_class(data, class_ratio = 0.05, seed=500):

    # Display old class counts
    print('The initial dataset has following sizes for each class:')
    print(data.click.value_counts())

    # Separate majority and minority classes
    data_majority = data[data.click == 0]
    data_minority = data[data.click == 1]

    print('Minority class is %.2f%% of initial sample size.' % (len(data_minority)/len(data)*100))

    # Samples to be drawn
    len_minority = math.floor(class_ratio / (1 - class_ratio) * len(data_majority))

    # Upsample
    data_minority_upsampled = resample(data_minority,
                                       replace=True,
                                       n_samples=len_minority,
                                       random_state=seed)

    # Combine minority class with downsampled majority class
    df_upsampled = pd.concat([data_majority, data_minority_upsampled])

    # Display new class counts
    print('New dataset has following sizes for each class:')
    print(df_upsampled.click.value_counts())
    print('Minority class is %.2f%% of total sample size.' % (len_minority/len(df_upsampled)*100))

    return df_upsampled

############################## END ##################################
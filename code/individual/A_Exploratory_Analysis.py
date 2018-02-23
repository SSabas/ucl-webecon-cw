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

#--------------------------------- GET DATA --------------------------------------#

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
validation = pd.read_csv('./data/validation.csv')

#--------------------------------- DECRIPTIVE ANALYSIS ---------------------------#

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)

# Some summary statistics
train.shape
train.describe().transpose()
train.info()

# Unique values
train["advertiser"].value_counts()
train["click"].value_counts() # Only 1793 clicks

## ANALYSIS BY ADVERTISER
train_by_advertiser = pd.DataFrame({'impressions': train.groupby('advertiser').size()}).reset_index() # Total Impressions
train_by_advertiser = train_by_advertiser.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('advertiser').size()}).reset_index(drop=True)) # Total Clicks
train_by_advertiser['CTR'] = train_by_advertiser['clicks']/train_by_advertiser['impressions']*100 # Click-Through Rate
train_by_advertiser = train_by_advertiser.join(pd.DataFrame({'cost': train.groupby(['advertiser'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_advertiser['CPM'] = train_by_advertiser['cost']*1000/train_by_advertiser['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_advertiser['eCPC'] = train_by_advertiser['cost']/train_by_advertiser['clicks'] # Effective Cost-per-Click


## ANALYSIS BY DAY/HOUR

## ANALYSIS OF SUCCESSFUL CLICKS

## ANALYSIS ON PRICES

# Difference between paid price and second best


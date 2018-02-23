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
import ggplot
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


## ANALYSIS BY DAY
train_by_weekday = pd.DataFrame({'impressions': train.groupby('weekday').size()}).reset_index() # Total Impressions
train_by_weekday = train_by_weekday.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('weekday').size()}).reset_index(drop=True)) # Total Clicks
train_by_weekday['CTR'] = train_by_weekday['clicks']/train_by_weekday['impressions']*100 # Click-Through Rate
train_by_weekday = train_by_weekday.join(pd.DataFrame({'cost': train.groupby(['weekday'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_weekday['CPM'] = train_by_weekday['cost']*1000/train_by_weekday['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_weekday['eCPC'] = train_by_weekday['cost']/train_by_weekday['clicks'] # Effective Cost-per-Click

## ANALYSIS BY HOUR
train_by_hour = pd.DataFrame({'impressions': train.groupby('hour').size()}).reset_index() # Total Impressions
train_by_hour = train_by_hour.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('hour').size()}).reset_index(drop=True)) # Total Clicks
train_by_hour['CTR'] = train_by_hour['clicks']/train_by_hour['impressions']*100 # Click-Through Rate
train_by_hour = train_by_hour.join(pd.DataFrame({'cost': train.groupby(['hour'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_hour['CPM'] = train_by_hour['cost']*1000/train_by_hour['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_hour['eCPC'] = train_by_hour['cost']/train_by_hour['clicks'] # Effective Cost-per-Click

## ANALYSIS OF SUCCESSFUL CLICKS
train_only_clicks = train[train['click'] == 1]
train_only_clicks.groupby('weekday').size()
train_only_clicks.groupby('hour').size()
train_only_clicks.groupby('advertiser').size()
train_only_clicks.groupby('weekday').size()

## ANALYSIS ON PRICES

# Histograms of prices
n, bins, patches = plt.hist(train['payprice'],  50, normed=1, facecolor='green', alpha=0.75, label='Paid Price')
n, bins, patches = plt.hist(train['bidprice'], 50, normed=1, facecolor='red', alpha=0.75, label='Bid Price')
n, bins, patches = plt.hist(train['slotprice'], 50, normed=1, facecolor='blue', alpha=0.75, label='Slot (Floor) Price')

plt.xlabel('Price (Chinese Fen)')
plt.ylabel('Probability')
plt.title('Histogram of Prices')
plt.grid(True)
plt.legend()

plt.show() # Quite a discrepancy in bid and paid prices

# Histogram of differences
n, bins, patches = plt.hist(train['bidprice']-train['payprice'],  50, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Price (Chinese Fen)')
plt.ylabel('Probability')
plt.title('Histogram of Difference between Bid Price and Paid Price')
plt.grid(True)
plt.legend()

plt.show() # Quite a discrepancy in bid and paid prices


############################## END ##################################


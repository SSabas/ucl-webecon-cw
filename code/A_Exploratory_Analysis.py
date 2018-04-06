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

# And own libraries
working_dir = os.getcwd() + ('/code')
sys.path.append(working_dir)
from B_Data_Preprocessing import *

#--------------------------------- GET DATA --------------------------------------#

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
validation = pd.read_csv('./data/validation.csv')

#--------------------------------- DESCRIPTIVE ANALYSIS ---------------------------#

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)

# Some summary statistics
train.shape
train.describe().transpose()
train.info()

# Unique values
train["advertiser"].value_counts()
train["click"].value_counts() # Only 1793 clicks

data.groupby('opsys', {'CTR':data.aggregate.average('click')})

# Unique values in each column
train.T.apply(lambda x: x.nunique(), axis=1)

## ANALYSIS BY ADVERTISER
train_by_advertiser = pd.DataFrame({'impressions': train.groupby('advertiser').size()}).reset_index() # Total Impressions
train_by_advertiser = train_by_advertiser.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('advertiser').size()}).reset_index(drop=True)) # Total Clicks
train_by_advertiser['CTR'] = (train_by_advertiser['clicks']/train_by_advertiser['impressions']*100).round(4) # Click-Through Rate
train_by_advertiser = (train_by_advertiser.join(pd.DataFrame({'cost': train.groupby(['advertiser'])['payprice'].sum()}).reset_index(drop=True)/1000)) # Total Pay/Cost
train_by_advertiser['CPM'] = (train_by_advertiser['cost']*1000/train_by_advertiser['impressions']).round(4) # Cost per-mille (Cost per Thousand Impressions)
train_by_advertiser['eCPC'] = (train_by_advertiser['cost']/train_by_advertiser['clicks']).round(4) # Effective Cost-per-Click
train_by_advertiser['']
train_by_advertiser.to_latex()


## ANALYSIS BY DAY
train_by_weekday = pd.DataFrame({'impressions': train.groupby('weekday').size()}).reset_index() # Total Impressions
train_by_weekday = (train_by_weekday.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('weekday').size()}).reset_index(drop=True))) # Total Clicks
train_by_weekday['CTR'] = (train_by_weekday['clicks']/train_by_weekday['impressions']*100) # Click-Through Rate
train_by_weekday = (train_by_weekday.join(pd.DataFrame({'cost': train.groupby(['weekday'])['payprice'].sum()}).reset_index(drop=True)/1000)) # Total Pay/Cost
train_by_weekday['CPM'] = (train_by_weekday['cost']*1000/train_by_weekday['impressions']).round(2) # Cost per-mille (Cost per Thousand Impressions)
train_by_weekday['eCPC'] = (train_by_weekday['cost']/train_by_weekday['clicks']).round(2) # Effective Cost-per-Click
train_by_weekday.to_latex()

CTR_vs_Weekday = train[['click', 'bidprice', 'payprice', 'advertiser', 'weekday']].groupby(
    ['advertiser', 'weekday']).sum()
CTR_vs_Weekday['impressions'] = train.groupby(['advertiser', 'weekday']).size()
CTR_vs_Weekday['CTR'] = CTR_vs_Weekday['click'] / CTR_vs_Weekday['impressions']
Plot_CTR_vs_Weekday = CTR_vs_Weekday.unstack('advertiser').loc[:, 'impressions'][[2997, 3358, 1458]]
Plot_CTR_vs_Weekday.fillna(0.0, inplace=True)  # Fill Nans with zero
Plot_CTR_vs_Weekday.plot(kind="line", color=['royalblue', 'darkred', 'darkgreen'])  # kind="bar"
ax = plt.gca()
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.xlabel('Weekday', fontsize=8)
plt.ylabel('Impressions', fontsize=8)
plt.xticks(rotation=0)
plt.title('Impressions Comparison based on Day of the Week', fontsize=10)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
plt.show()
plt.savefig(os.getcwd()+'/results/line_hourly.pdf')


## ANALYSIS BY HOUR
train_by_hour = pd.DataFrame({'impressions': train.groupby('hour').size()}).reset_index() # Total Impressions
train_by_hour = train_by_hour.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('hour').size()}).reset_index(drop=True)) # Total Clicks
train_by_hour['CTR'] = train_by_hour['clicks']/train_by_hour['impressions']*100 # Click-Through Rate
train_by_hour = train_by_hour.join(pd.DataFrame({'cost': train.groupby(['hour'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_hour['CPM'] = train_by_hour['cost']*1000/train_by_hour['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_hour['eCPC'] = train_by_hour['cost']/train_by_hour['clicks'] # Effective Cost-per-Click

CTR_vs_Hour = train[['click', 'bidprice', 'payprice', 'advertiser', 'hour']].groupby(['advertiser', 'hour']).sum()
CTR_vs_Hour['impressions'] = train.groupby(['advertiser', 'hour']).size()
CTR_vs_Hour['CTR'] = CTR_vs_Hour['click'] / CTR_vs_Hour['impressions']
Plot_CTR_vs_Hour = CTR_vs_Hour.unstack('advertiser').loc[:, 'click'][[2997, 3358, 1458]]
Plot_CTR_vs_Hour.fillna(0.0, inplace=True)  # Fill Nans with zero
Plot_CTR_vs_Hour.plot(kind="line", color=['royalblue', 'darkred', 'forestgreen'])  # kind="bar"
plt.xlabel('Hour', fontsize=8)
plt.ylabel('Clicks', fontsize=8)
plt.title('Clicks Comparison based on Hour of the Day', fontsize=10)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
plt.show()
plt.savefig(os.getcwd()+'/results/line_clicks.pdf')


## ANALYSIS BY OP SYSTEM
train = separate_useragent(train)
train_by_op_sys = pd.DataFrame({'impressions': train.groupby('opsys').size()}).reset_index() # Total Impressions
train_by_op_sys = train_by_op_sys.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('opsys').size()}).reset_index(drop=True)) # Total Clicks
train_by_op_sys['CTR'] = train_by_op_sys['clicks']/train_by_op_sys['impressions']*100 # Click-Through Rate
train_by_op_sys = train_by_op_sys.join(pd.DataFrame({'cost': train.groupby(['hour'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_op_sys['CPM'] = train_by_op_sys['cost']*1000/train_by_op_sys['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_op_sys['eCPC'] = train_by_op_sys['cost']/train_by_op_sys['clicks'] # Effective Cost-per-Click

CTR_vs_OS = train[['click', 'bidprice', 'payprice', 'advertiser', 'opsys']].groupby(['advertiser', 'opsys']).sum()
CTR_vs_OS['impressions'] = train.groupby(['advertiser', 'opsys']).size()
CTR_vs_OS['CTR'] = CTR_vs_OS['click'] / CTR_vs_OS['impressions']
Plot_CTR_vs_OS = CTR_vs_OS.unstack('advertiser').loc[:, 'CTR'][[2997, 3358, 1458, 2259]]
Plot_CTR_vs_OS.fillna(0.0, inplace=True)  # Fill Nans with zero
Plot_CTR_vs_OS.plot(kind="bar", color=['royalblue', 'darkred', 'forestgreen', 'darkorange'])  # kind="bar"
plt.xlabel('`Operating System', fontsize=8)
plt.ylabel('CTR', fontsize=8)
plt.xticks(rotation=0)
plt.title('CTR Comparison based on Operating System', fontsize=10)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
plt.savefig(os.getcwd()+'/results/barplot.pdf')


## ANALYSIS BY BROWSER
train = separate_useragent(train)
train_by_browser = pd.DataFrame({'impressions': train.groupby('browser').size()}).reset_index() # Total Impressions
train_by_browser = train_by_browser.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('browser').size()}).reset_index(drop=True)) # Total Clicks
train_by_browser['CTR'] = train_by_browser['clicks']/train_by_browser['impressions']*100 # Click-Through Rate
train_by_browser = train_by_browser.join(pd.DataFrame({'cost': train.groupby(['hour'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_browser['CPM'] = train_by_browser['cost']*1000/train_by_browser['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_browser['eCPC'] = train_by_browser['cost']/train_by_browser['clicks'] # Effective Cost-per-Click

# ANALYSIS BY SLOTS
train = slot_width_height_combiner(train)
train_by_slot = pd.DataFrame({'impressions': train.groupby('slot_width_height').size()}).reset_index() # Total Impressions
train_by_slot = train_by_slot.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('slot_width_height').size()}).reset_index(drop=True)) # Total Clicks
train_by_slot['CTR'] = train_by_slot['clicks']/train_by_slot['impressions']*100 # Click-Through Rate
train_by_slot = train_by_slot.join(pd.DataFrame({'cost': train.groupby(['hour'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_slot['CPM'] = train_by_slot['cost']*1000/train_by_slot['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_slot['eCPC'] = train_by_slot['cost']/train_by_slot['clicks'] # Effective Cost-per-Click

# BY PRICE BUCKET
train['slotprice_binned'] = train['slotprice'].apply(slot_price_bucketing)
train_by_slotprice = pd.DataFrame({'impressions': train.groupby('slotprice_binned').size()}).reset_index() # Total Impressions
train_by_slotprice = train_by_slotprice.join(pd.DataFrame({'clicks': train[train['click'] == 1].groupby('slotprice_binned').size()}).reset_index(drop=True)) # Total Clicks
train_by_slotprice['CTR'] = train_by_slotprice['clicks']/train_by_slotprice['impressions']*100 # Click-Through Rate
train_by_slotprice = train_by_slotprice.join(pd.DataFrame({'cost': train.groupby(['hour'])['payprice'].sum()}).reset_index(drop=True)/1000) # Total Pay/Cost
train_by_slotprice['CPM'] = train_by_slotprice['cost']*1000/train_by_slotprice['impressions'] # Cost per-mille (Cost per Thousand Impressions)
train_by_slotprice['eCPC'] = train_by_slotprice['cost']/train_by_slotprice['clicks'] # Effective Cost-per-Click

## ANALYSIS OF SUCCESSFUL CLICKS
train_only_clicks = train[train['click'] == 1]
train_only_clicks.groupby('weekday').size()
train_only_clicks.groupby('hour').size()
train_only_clicks.groupby('advertiser').size()
train_only_clicks.groupby('weekday').size()

## ANALYSIS ON PRICES

# Histograms of prices
n, bins, patches = plt.hist(train['bidprice'],  50, normed=1, facecolor='green', alpha=0.25, label='Paid Price')
n, bins, patches = plt.hist(validation['bidprice'], 50, normed=1, facecolor='red', alpha=0.25, label='Bid Price')
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


# Violin plots
# Turn columns to numpy arrays
df_all = np.array(train[['bidprice', 'payprice', 'slotprice']])
train_clicks = train[train['click'] == 1]
df_clicks = np.array(train_clicks[['bidprice', 'payprice', 'slotprice']])


# Define some plotting functions
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out', labelsize = 8)
    ax.get_yaxis().set_tick_params(labelsize = 8)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    # ax.set_xlabel('Sample name', fontsize=8)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.3*1.2, 2.2*1.2*2), sharey=True)

ax1.set_title('Full Dataset (2,430,981 Observations)', fontsize=10)
ax1.set_ylabel('Prices (CNY Fen)', fontsize=8)
parts = ax1.violinplot(
        df_all, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('royalblue')
    pc.set_edgecolor('')
    pc.set_alpha(0.75)

ax2.set_title('Clicks Only Dataset (1,793 Observations)', fontsize=10)
ax2.set_ylabel('Prices (CNY Fen)', fontsize=8)

parts2 = ax2.violinplot(
        df_clicks, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts2['bodies']:
    pc.set_facecolor('darkred')
    pc.set_edgecolor('')
    pc.set_alpha(0.75)


quartile1, medians, quartile3 = np.percentile(df_clicks, [25, 50, 75], axis=0)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(df_all, quartile1, quartile3)])
whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

quartile1, medians, quartile3 = np.percentile(df_all, [25, 50, 75], axis=0)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(df_all, quartile1, quartile3)])
whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

# set style for the axes
labels = ['Bid Price', 'Pay Price', 'Floor Price']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(top=0.94, bottom=0.05, hspace=0.20)
plt.show()
plt.savefig(os.getcwd()+'/results/violinplot.pdf')

# CORRELATION OF PRICES
plt.style.use("seaborn-whitegrid")
params = {'legend.fontsize': '6',
          'figure.figsize': (3.3*1.2, 2.2*1.2),
          'axes.labelsize': '8',
          'axes.titlesize': '10',
          'xtick.labelsize': '6',
          'ytick.labelsize': '6'}
sns.set(font_scale=6/8)


# Correlate the prices
df_corr = train[['bidprice', 'payprice', 'slotprice']].corr()
df_corr.columns = [['Bid Price', 'Pay Price', 'Floor Price']]
corr = df_corr.corr().round(2)


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(3.3*1.2, 2.2*1.2))
plt.tick_params(labelsize=6)
plt.title('Correlogram of Prices', fontsize=10)
plt.xticks(rotation=0)
plt.legend(fontsize=6)
plt.xticks(rotation=0)

# Generate a custom diverging colormap
sns.set_style("whitegrid")
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(os.getcwd()+'/results/correlogram.pdf')

# 3 SUBPLOT GRAPH FOR GROUP REPORT
import matplotlib as mpl
mpl.style.use('seaborn-whitegrid')

params = {'legend.fontsize': '6',
          'figure.figsize': (3.3*1.2, 2.2*1.2),
          'axes.labelsize': '8',
          'axes.titlesize': '10',
          'xtick.labelsize': '6',
          'ytick.labelsize': '6'}

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(3.3*1.2, 2.2*3.6), sharey=False)
plt.style.use("seaborn-darkgrid")
# with sns.axes_style("darkgrid"):
#     ax1 = fig.add_subplot()
#     ax2 = fig.add_subplot()
#     ax3 = fig.add_subplot()


fig = plt.figure(figsize=(3.3*1.2, 2.2*1.2))
with sns.axes_style("darkgrid"):
    ax1 = fig.add_subplot(111)
CTR_vs_Weekday = train[['click', 'bidprice', 'payprice', 'advertiser', 'weekday']].groupby(['advertiser', 'weekday']).sum()
CTR_vs_Weekday['impressions'] = train.groupby(['advertiser', 'weekday']).size()
CTR_vs_Weekday['CTR'] = CTR_vs_Weekday['click'] / CTR_vs_Weekday['impressions']
Plot_CTR_vs_Weekday = CTR_vs_Weekday.unstack('advertiser').loc[:, 'CTR'][[1458, 3358]]
Plot_CTR_vs_Weekday.fillna(0.0, inplace=True)  # Fill Nans with zero
Plot_CTR_vs_Weekday.plot(kind="line", color=['royalblue', 'darkred', 'darkgreen'], ax=ax1)  # kind="bar"
ax1.set_title('CTR vs Weekday', fontsize=10)
ax1.set_ylabel('CTR', fontsize=8)
ax1.set_xlabel('Weekday', fontsize=8)


# OP system
CTR_vs_OS = train[['click', 'bidprice', 'payprice', 'advertiser', 'opsys']].groupby(['advertiser', 'opsys']).sum()
CTR_vs_OS['impressions'] = train.groupby(['advertiser', 'opsys']).size()
CTR_vs_OS['CTR'] = CTR_vs_OS['click'] / CTR_vs_OS['impressions']
Plot_CTR_vs_OS = CTR_vs_OS.unstack('advertiser').loc[:, 'CTR'][[1458, 3358]]
Plot_CTR_vs_OS.fillna(0.0, inplace=True)  # Fill Nans with zero
plt.style.use("seaborn-whitegrid")
Plot_CTR_vs_OS.plot(kind="bar", color=['royalblue', 'darkred', 'forestgreen', 'darkorange'], ax=ax2)  # kind="bar"
ax2.set_title('CTR vs Operating System', fontsize=10)
ax2.set_ylabel('CTR', fontsize=8)
ax2.set_xlabel('Operating System', fontsize=8)
ax2.xaxis.set_tick_params(rotation=0)

# Ad Exchange
CTR_vs_ad = train[['click', 'bidprice', 'payprice', 'advertiser', 'adexchange']].groupby(['advertiser', 'adexchange']).sum()
CTR_vs_ad['impressions'] = train.groupby(['advertiser', 'adexchange']).size()
CTR_vs_ad['CTR'] = CTR_vs_ad['click'] / CTR_vs_ad['impressions']
Plot_CTR_vs_ad = CTR_vs_ad.unstack('advertiser').loc[:, 'CTR'][[1458, 3358]]
Plot_CTR_vs_ad.fillna(0.0, inplace=True)  # Fill Nans with zero
Plot_CTR_vs_ad.plot(kind="bar", color=['royalblue', 'darkred', 'forestgreen', 'darkorange'], ax=ax3)  # kind="bar"
ax3.set_title('CTR vs Ad Exchange', fontsize=10)
ax3.set_ylabel('CTR', fontsize=8)
ax3.set_xlabel('Ad Exchange', fontsize=8)
ax3.xaxis.set_tick_params(rotation=0)

plt.subplots_adjust(top=0.96, bottom=0.05, hspace=0.30, left = 0.2)
plt.show()
plt.savefig(os.getcwd()+'/results/adv_sample2.pdf', dpi = 300)

############################## END ##################################
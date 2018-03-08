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
import numpy as np


# --------------------------------- FITTING --------------------------------------- #

# --- CONSTANT BIDDING STRATEGY
def constant_bidding_strategy(data, constant, budget=625000):

    # Get boolean vector of the bids won
    bids_won = np.array(data['bidprice']) < np.repeat(constant, len(data['bidprice']))

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['bidprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned

def random_bidding_strategy(data, lower_bound=0, upper_bound=400, budget=625000, seed=500):

    # Generate bids
    bids = np.random.randint(lower_bound, upper_bound, len(data))

    # Get boolean vector of the bids won
    bids_won = np.array(data['bidprice']) < np.repeat(bids, len(data['bidprice']))

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['bidprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned

random_bidding_strategy(data, lower_bound=0, upper_bound=400, budget=625000, seed=500)

constant_bidding_strategy(validation1, 120, budget=625000)

    # Initialise output
    impression = 0 # Bids won
    clicks = 0 # Number of clicks for the bids won
    cost = 0.0 # Accumulative cost

    for click, pay_price in dataset[['click', 'payprice']].values:
        if constant > pay_price:
            impression += 1
            clicks += click
            cost += pay_price
        if cost >= budget:
            break

    return impression, clicks, cost


# --- RANDOM BIDDING STRATEGY
def random_bidding(dataset, upper_bound, budget = 625000):

    # Initialise output
    impression = 0.0
    clicks = 0
    cost = 0.0

    for click, pay_price in validation[['click', 'payprice']].values:
        rand_no = randrange(upper_bound)
        if rand_no > pay_price:
            impression += 1
            clicks += click
            cost += pay_price
        if cost >= budget:
            break

    return impression, clicks, cost

# --- pCTR BASED BIDDING STRATEGIES (CRUDE PARAMETER ESTIMATION)


# Average CTR
avgCTR = (train.click.sum() / train.logtype.sum())

avgCTR = 0.08
# Bid generator
def bid_gen(strat_type, lowerbound, upperbound, step):
    bids = []
    base_bids = np.arange(lower_bound, upper_bound, step)

    for base_bid in base_bids:
        for i in range(0, len(pred)):
            bid = base_bid * (pred[i] / avgCTR)
            bids.append(bid)

            if strat_type == 'linear':
                bid = base_bid * (pred[i] / avgCTR)
                bids.append(bid)
            elif strat_type == 'nonlinear':
                bid = base_bid * (pred[i] / avgCTR) ** 2
                bids.append(bid)
            elif strat_type == 'exponential':
                bid = base_bid * np.exp(pred[i] / avgCTR)
                bids.append(bid)
    bid_chunks = [bids[x:x + len(pred)] for x in range(0, len(bids), len(pred))]
    return bid_chunks, base_bids


# Bidding function
def bidding(bids):
    impression = 0.0
    clicks = 0
    cost = 0.0
    budget = 6250000

    bool_check = bids >= validation.payprice
    for i in range(0, len(bool_check)):
        if bool_check[i] == True:
            impression += 1.0
            clicks += validation.click[i]
            cost += validation.payprice[i]
        if cost >= budget:
            break
    return impression, clicks, cost


# Apply different bidding strategies
def bid_strategy(strat):
    startbidgentime = time.time()

    print("Starting %s bid generation" % strat)
    bid_chunks, base_bids = bid_gen(strat, 2, 302, 2)
    print(" %s big generation finished in %s minutes" % (strat, round(((time.time() - startbidgentime) / 60), 2)))

    bid_df = pd.DataFrame()
    bid_df['bid'] = base_bids
    bid_df['bidding_strategy'] = strat

    im = []
    clks = []
    ct = []

    startbidtime = time.time()
    print("Starting %s Bidding" % strat)

    for bids in bid_chunks:
        [imps, clicks, cost] = bidding(bids)
        im.append(imps)
        clks.append(clicks)
        ct.append(cost)

    bid_df['imps_won'] = im
    bid_df.imps_won = bid_df.imps_won.astype(int)
    bid_df['total_spend'] = ct
    bid_df['clicks'] = clks
    bid_df['CTR'] = (bid_df.clicks / bid_df.imps_won * 100).round(4).astype(str)
    bid_df['CPM'] = (bid_df.total_spend / bid_df.imps_won * 1000).round(2).astype(str)
    bid_df['CPC'] = (bid_df.total_spend / bid_df.clicks).round(2).astype(str)
    print(" %s bidding finished in %s minutes" % (strat, round(((time.time() - startbidtime) / 60), 2)))
    return bid_df

# ---
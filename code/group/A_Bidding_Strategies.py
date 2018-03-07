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
def constant_bidding_strategy(click_data, payprice_data, constant, budget = 625000):

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

# --- pCTR BASED BIDDING STRATEGIES (CRUDE ESTIMATION

# ---
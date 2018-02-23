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


#--------------------------------- FITTING ---------------------------------------#

# Logistics regression
# Decision trees
# Field-aware Factorization Machines

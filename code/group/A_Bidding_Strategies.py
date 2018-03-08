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
import time


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
    bids_won = np.array(data['bidprice']) < bids

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['bidprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


# --- pCTR BASED BIDDING STRATEGIES (CRUDE PARAMETER ESTIMATION)
def parametrised_bidding_strategy(data, prediction, type = 'linear', parameter = 100, budget = 625000):

    # Compute average CTR as a base metric
    avgCTR = np.repeat(np.sum(data['click'] == 1) / data.shape[0], prediction.shape[0])

    # Calculate bids based on the model

    # For linear model
    if type == 'linear':
        bids = np.repeat(parameter, prediction.shape[0]) * (np.array(prediction) + avgCTR)

    if type == 'square':
        bids = np.repeat(parameter, prediction.shape[0]) * (np.array(prediction) / avgCTR) ** 2

    if type == 'exponential':
        bids = np.repeat(parameter, prediction.shape[0]) * np.exp(np.array(prediction) / avgCTR)

    # Get boolean vector of the bids won
    bids_won = np.array(data['bidprice']) < bids

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['bidprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


# --- Optimal Real Time Bidding (ORTB)
def ORTB_strategy(data, prediction, type = 'ORTB1', c=50, b=1, budget=625000):

    """
    ORTB1 formula:
    sqrt(c/lambda * pctr + c**2) - c

    ORTB2 formula:
    term = (pCTR + np.sqrt(c * c * lambda_ * lambda_ + pCTR * pCTR)) / (c * lambda_)
    bid = c * (term**(1 / 3) - term**(-1 / 3))
    """

    # Calculate bids based on the specified model
    size = prediction.shape[0]

    if type == 'ORTB1':
        bids = np.sqrt(np.repeat(c, size) / np.repeat(b, size) * np.array(prediction) + np.repeat(c, size) ** 2) \
               - np.repeat(c, size)
    else:
        term = (np.array(prediction) + np.sqrt(np.repeat(c, size) ** 2
                                               * np.repeat(b, size) * 2 + np.repeat(c, size) ** 2)) / \
               (np.repeat(c, size) * np.repeat(b, size))
        bids = np.repeat(c, size) * (term ** (1 / 3) - term ** (-1 / 3))

    # Get boolean vector of the bids won
    bids_won = np.array(data['bidprice']) < bids

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['bidprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


ORTB_strategy(data, prediction, type = 'ORTB2', c=50, b=1, budget=625000)


# --- Evaluate Strategies Using Different Parameter Combinations
def strategy_evaluation(data, prediction, parameter_range, type = 'linear',  budget = 625000,
                        only_best = 'no', to_plot = 'yes', repeated_runs = 1):

    # Time it
    start_time = time.time()

    # Initialise output
    colnames = ['type', 'budget', 'parameter_1', 'parameter_2', 'total_auctions',
                'ads_auctioned_for', 'impressions_won', 'clicks_won',
                'CTR', 'CPM', 'CPC']
    output = pd.DataFrame(index=range(len(parameter_range)), columns=colnames)

    for i, parameter in zip(range(len(parameter_range)), parameter_range):

        if len(parameter_range[0]) == 1:
            output['parameter_1'][i] = parameter

        else:
            output['parameter_1'][i] = parameter[0]
            output['parameter_2'][i] = parameter[1]

        if type == 'constant':

            output['impressions_won'][i], \
            output['clicks_won'][i], \
            output['ads_auctioned_for'][i] = \
                constant_bidding_strategy(data, parameter, budget=budget)

        elif type == 'random':

            impressions_won = []
            clicks_won = []
            ads_auctioned_for = []

            for run in range(0, repeated_runs):

                impressions, clicks, ads_auctioned = random_bidding_strategy(data, parameter[0],
                                                                             parameter[1], budget=budget)
                impressions_won.append(impressions)
                clicks_won.append(clicks)
                ads_auctioned_for.append(ads_auctioned)

            output['impressions_won'][i] = np.mean(impressions_won)
            output['clicks_won'][i] = np.mean(clicks_won)
            output['ads_auctioned_for'][i] = np.mean(ads_auctioned_for)

        elif type[0:4] == 'ORTB':

            # Iterate over the parameter range and complete the table
            output['impressions_won'][i], \
            output['clicks_won'][i], \
            output['ads_auctioned_for'][i] = \
                ORTB_strategy(data, prediction, type=type, c=parameter[0], b=parameter[1], budget=625000)

        else:

            # Iterate over the parameter range and complete the table
            output['impressions_won'][i], \
            output['clicks_won'][i], \
            output['ads_auctioned_for'][i] = \
                parametrised_bidding_strategy(data, prediction, type=type, parameter=parameter, budget=budget)

    # Fill in last columns
    output['type'] = type
    output['budget'] = budget
    output['total_auctions'] = prediction.shape[0]
    output['CTR'] = output['clicks_won']/ output['impressions_won']
    output['CPM'] = output['budget']/ output['impressions_won'] * 1000
    output['CPC'] = output['budget']/ output['clicks_won']

    print("Evaluation for %s type model finished in %.2f seconds." % (type, (time.time() - start_time)/360))

    if to_plot == 'yes':

        # Set title and style
        plot_title = "Performance evaluation of %s model"% (type)
        plt.style.use("seaborn-darkgrid")

        # Plot bidding performance
        fig, ax1 = plt.subplots()
        ax1.plot(output.parameter_1, output.clicks_won, marker='o', markersize =2, color = 'royalblue', label='Clicks')
        ax1.set_xlabel('Model Parameter')
        ax1.set_ylabel('Clicks Won', color='royalblue')
        ax1.set_title(plot_title, fontsize=12)
        ax1.axvline(x=output.parameter_1[output.clicks_won.argmax()],
                    ymax=1, linewidth=1, color='royalblue', linestyle='--',
                    label='Parameter with Max Clicks')

        ax2 = ax1.twinx()
        ax2.plot(output.parameter_1, output.CTR, marker='s', markersize =2, color='darkred', label='CTR')
        ax2.set_ylabel('CTR', color='darkred')
        ax2.axvline(x=output.parameter_1[output.CTR.argmax()],
                    ymax=1, linewidth=1, color='darkred', linestyle='--',
                    label='Parameter with Max CTR')

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc='best', frameon=True)

        plt.show()

    return output

strategy_evaluation(validation1, prediction, parameter_range = np.linspace(200, 325, 100), type = 'random',  budget = 625000,
                        only_best = 'no', to_plot = 'yes')

strategy_evaluation(validation1, prediction, parameter_range = [[100,400], [200,500]], type = 'random',  budget = 625000,
                        only_best = 'no', to_plot = 'no', repeated_runs = 20)

strategy_evaluation(validation1, prediction, parameter_range = [[100,400], [200,500]], type = 'random',  budget = 625000,
                        only_best = 'no', to_plot = 'yes', repeated_runs = 20)

a =
b = np.linspace(5e-7, 1, 1000)
a = np.repeat(1, 50 100)


a = np.linspace(1, 50, 100)
b = np.repeat(5e-7, len(a))


strategy_evaluation(validation1, prediction, parameter_range = np.column_stack((a, b)), type = 'ORTB1',  budget = 625000,
                        only_best = 'no', to_plot = 'yes', repeated_runs = 20)

a = [1,2,3]
b = [4,5,6]
np.column_stack((a, b))


# --- Optimal Real Time Bidding (ORTB)

# --- Max eCPC

# --- Gate Bidding
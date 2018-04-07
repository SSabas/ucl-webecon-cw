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
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
import matplotlib.cm as cm


# ----------------------------- BID NORMALISATION --------------------------------- #
def normalise_bids(prediction, minority_weighting = 0.025):
    '''
    Formula (taken from FB paper)
    '''

    # Normalise the prediction based on the formula above

    output = prediction / (prediction + (1-prediction)/np.repeat(minority_weighting, prediction.shape[0]))

    return output


# --------------------------------- FITTING --------------------------------------- #

# --- CONSTANT BIDDING STRATEGY
def constant_bidding_strategy(data, constant, budget=6250000):

    # Get boolean vector of the bids won
    bids_won = np.array(data['payprice']) < np.repeat(constant, len(data['bidprice']))

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['payprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


def random_bidding_strategy(data, lower_bound=0, upper_bound=400, budget=6250000):

    # Generate bids
    bids = np.random.randint(lower_bound, upper_bound, len(data))

    # Get boolean vector of the bids won
    bids_won = np.array(data['payprice']) < bids

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['payprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


# --- pCTR BASED BIDDING STRATEGIES (CRUDE PARAMETER ESTIMATION)
def parametrised_bidding_strategy(data, prediction, type='linear', parameter=100, budget=625000,
                                  average_CTR=None):

    if average_CTR == None:
        avgCTR = np.repeat(average_CTR, prediction.shape[0])

    else:
        # Calculate bids based on the model
        avgCTR = np.repeat(np.sum(data['click'] == 1) / data.shape[0], prediction.shape[0])

    # For linear model
    if type == 'linear':
        bids = np.repeat(parameter, prediction.shape[0]) * (np.array(prediction) / avgCTR)

    if type == 'square':
        bids = np.repeat(parameter, prediction.shape[0]) * (np.array(prediction) / avgCTR) ** 2

    if type == 'exponential':
        bids = np.repeat(parameter, prediction.shape[0]) * np.exp(np.array(prediction) / avgCTR)

    # Get boolean vector of the bids won
    bids_won = np.array(data['payprice']) < bids

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['payprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


# --- Optimal Real Time Bidding (ORTB)
def ORTB_strategy(data, prediction, type = 'ORTB1', c=50, b=1, budget=6250000, average_CTR=7.375623e-04):

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

    elif type == 'ORTBx':
        bids = (np.array(prediction) / np.repeat(average_CTR, size)) ** 2 * c + b

    elif type == 'ORTBy':
        bids = (np.array(prediction) / np.repeat(average_CTR, size)) ** 2 * c + (np.array(prediction) /
                                                                                 np.repeat(average_CTR, size)) * b

    else:
        term = (np.array(prediction) + np.sqrt(np.repeat(c, size) ** 2
                                               * np.repeat(b, size) * 2 + np.array(prediction) ** 2)) \
            / (np.repeat(c, size) * np.repeat(b, size))
        bids = np.repeat(c, size) * ((term ** (1 / 3)) - (term ** (-1 / 3)))

    # Get boolean vector of the bids won
    bids_won = np.array(data['payprice']) < bids

    # Get cumulative sum conditional on the win
    bids_won_cumsum = np.cumsum(np.array(data['payprice'])*bids_won)

    # Get a boolean vector where bids cumsum is still under budget limit
    valid_bids = bids_won_cumsum <= np.repeat(budget, len(bids_won_cumsum))

    # Get evaluation metrics
    impressions = np.sum(valid_bids * bids_won)
    clicks = np.sum(valid_bids * bids_won * (np.array(data['click']) == 1))
    ads_auctioned = np.sum(bids_won_cumsum<budget)

    return impressions, clicks, ads_auctioned


# --- Evaluate Strategies Using Different Parameter Combinations
def strategy_evaluation(data, prediction, parameter_range, type = 'linear',  budget = 6250000,
                        only_best = 'no', to_plot = 'yes', plot_3d = 'no', repeated_runs = 1,
                        average_CTR = 7.375623e-04, to_save='no', file_name='bidding_strategy.pdf'):

    # Time it
    start_time = time.time()

    # Initialise output
    colnames = ['type', 'budget', 'parameter_1', 'parameter_2', 'total_auctions',
                'ads_auctioned_for', 'impressions_won', 'clicks_won',
                'CTR', 'CPM', 'CPC']
    output = pd.DataFrame(index=range(len(parameter_range)), columns=colnames)

    for i, parameter in zip(range(len(parameter_range)), parameter_range):

        print(i, parameter)

        if parameter_range[0].size == 1:
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
                ORTB_strategy(data, prediction, type=type, c=parameter[0], b=parameter[1], budget=budget,
                              average_CTR=average_CTR)

        else:

            # Iterate over the parameter range and complete the table
            output['impressions_won'][i], \
            output['clicks_won'][i], \
            output['ads_auctioned_for'][i] = \
                parametrised_bidding_strategy(data, prediction, type=type, parameter=parameter, budget=budget,
                                              average_CTR = average_CTR)

    # Fill in last columns
    output['type'] = type
    output['budget'] = budget
    output['total_auctions'] = prediction.shape[0]
    output['CTR'] = output['clicks_won']/ output['impressions_won']
    output['CPM'] = output['budget']/ output['impressions_won'] * 1000
    output['CPC'] = output['budget']/ output['clicks_won']

    print("Evaluation for %s type model finished in %.2f seconds." % (type, (time.time() - start_time)))

    if to_plot == 'yes':

        if plot_3d == 'yes':

            # Get the grid for clicks
            x1_clicks = np.linspace(output['parameter_1'].min(), output['parameter_1'].max(),
                             len(output['parameter_1'].unique()))
            y1_clicks = np.linspace(output['parameter_2'].min(), output['parameter_2'].max(),
                             len(output['parameter_2'].unique()))
            x2_clicks, y2_clicks = np.meshgrid(x1_clicks, y1_clicks)
            z2_clicks = griddata((output['parameter_1'], output['parameter_2']), output['clicks_won'], (x2_clicks, y2_clicks),
                          method='linear')

            # Get the grid for CTR
            x1_CTR = np.linspace(output['parameter_1'].min(), output['parameter_1'].max(),
                             len(output['parameter_1'].unique()))
            y1_CTR = np.linspace(output['parameter_2'].min(), output['parameter_2'].max(),
                             len(output['parameter_2'].unique()))
            x2_CTR, y2_CTR = np.meshgrid(x1_CTR, y1_CTR)
            z2_CTR = griddata((output['parameter_1'], output['parameter_2']), output['CTR'], (x2_CTR, y2_CTR),
                          method='linear')

            # Set the parameters for plotting
            plt.style.use("seaborn-whitegrid")
            params = {'legend.fontsize': '8',
                      'figure.figsize': (20,10),
                      'axes.labelsize': '8',
                      'axes.titlesize': '12',
                      'xtick.labelsize': '6',
                      'ytick.labelsize': '6'}

            plt.rcParams.update(params)
            plot_title = "Performance Evaluation of %s Model"%(type)

            # Plot the first subplot
            fig = plt.figure(figsize=(3.3*1.2, 2.2*1.2*2))
            ax = fig.add_subplot(2, 1, 1, projection='3d')
            surf = ax.plot_surface(x2_clicks, y2_clicks, z2_clicks, rstride=1, cstride=1, cmap=cm.Blues,
                                   linewidth=0.01, antialiased=False, edgecolors='grey', alpha=0.8)

            tmp_planes = ax.zaxis._PLANES
            ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                                tmp_planes[0], tmp_planes[1],
                                tmp_planes[4], tmp_planes[5])
            view_1 = (25, -135)
            view_2 = (25, -45)
            init_view = view_2
            ax.view_init(*init_view)

            ax.set_xlabel('Parameter 1', fontsize = 6)
            ax.set_ylabel('Parameter 2', fontsize = 6)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel('Clicks', fontsize = 6, rotation = 90)
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.major.formatter._useMathText = True
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.title('Clicks', fontsize = 10)

            # Plot the second subplot
            ax = fig.add_subplot(2, 1, 2, projection='3d')
            surf = ax.plot_surface(x2_CTR, y2_CTR, z2_CTR, rstride=1, cstride=1, cmap=cm.Reds,
                                   linewidth=0.01, antialiased=False, edgecolors='grey', alpha=0.8)

            tmp_planes = ax.zaxis._PLANES
            ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                                tmp_planes[0], tmp_planes[1],
                                tmp_planes[4], tmp_planes[5])
            view_1 = (25, -135)
            view_2 = (25, -45)
            init_view = view_2
            ax.view_init(*init_view)

            ax.set_xlabel('Parameter 1', fontsize = 6)
            ax.set_ylabel('Parameter 2', fontsize = 6)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel('CTR', fontsize = 6, rotation = 90)
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.major.formatter._useMathText = True
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.title('CTR', fontsize = 10)
            plt.subplots_adjust(top=0.94, bottom=0.05, hspace=0.20)
            plt.suptitle(plot_title, fontsize = 10)

        else:

            # Set title and style
            # plot_title = "Performance evaluation of %s model"%(type)
            plot_title = "Performance of Linear Bidding Strategy"
            plt.style.use("seaborn-darkgrid")

            # Plot bidding performance
            fig, ax1 = plt.subplots()
            ax1.plot(output.parameter_1, output.clicks_won, marker='o', markersize =1, color = 'royalblue', label='Clicks')
            ax1.set_xlabel('Model Parameter', fontsize = 8)
            ax1.set_ylabel('Clicks Won', color='royalblue', fontsize = 8)
            ax1.set_title(plot_title, fontsize=10)
            ax1.ticklabel_format(fontsize=6)
            ax1.vlines(x=output.parameter_1[output.clicks_won.argmax()], ymin=0,
                       ymax=output.clicks_won[output.clicks_won.argmax()], linewidth=1, color='royalblue', linestyle='--',
                       label='Parameter with Max Clicks')
            ax1.hlines(xmin=output.parameter_1[output.parameter_1.argmin()], xmax=output.parameter_1[output.clicks_won.argmax()],
                       y=output.clicks_won[output.clicks_won.argmax()], linewidth=1, color='royalblue', linestyle='--',
                       label='Parameter with Max Clicks')
            ax1.set_axisbelow(True) # Push gridlines back
            ax1.tick_params(axis='x', labelsize=6)
            ax1.tick_params(axis='y', labelsize=6)


            ax2 = ax1.twinx()
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0), fontsize=6)
            ax2.yaxis.offsetText.set_fontsize(6)
            ax2.tick_params(axis='y', labelsize=8)
            # ax2.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
            ax2.plot(output.parameter_1, output.CTR, marker='s', markersize =1, color='darkred', label='CTR')
            ax2.set_ylabel('CTR', color='darkred', fontsize = 8)
            ax2.set_axisbelow(True)
            ax2.grid(None)

            # ax2.vlines(x=output.parameter_1[output.CTR.argmax()], ymin=0,
            #            ymax=output.CTR[output.CTR.argmax()], linewidth=1, color='darkred', linestyle='--',
            #            label='Parameter with Max CTR')
            # ax2.hlines(xmin=output.parameter_1[output.parameter_1.argmin()], xmax=output.parameter_1[output.CTR.argmax()],
            #            y=output.CTR[output.CTR.argmax()], linewidth=1, color='darkred', linestyle='--',
            #            label='Parameter with Max CTR')

            lines = ax1.get_lines() + ax2.get_lines()
            ax1.legend(lines, [line.get_label() for line in lines], loc='top right', frameon=True)

        if to_save =='yes':

            fig.set_size_inches(3.3*1.2, 2.2*1.2)
            fig.savefig(os.getcwd() + '/results/' + file_name, dpi=300)

        plt.show()

    return output

####################### END ########################
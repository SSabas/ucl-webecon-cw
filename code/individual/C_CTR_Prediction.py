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
import xgboost
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from fastFM import als
import scipy.sparse as sp
import pickle


# --------------------------------- FITTING --------------------------------------- #


# --- PLOT ROC CURVE
def plot_ROC_curve(data, prediction):
    """
    Function to plot the ROC curve with AUC.
    """

    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(data, prediction)
    roc_auc = roc_auc_score(data, prediction)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")


# --- LOGISTIC REGRESSION
def logistic_model(train, validation,
                   parameters = {'C': [0.1, 1, 2, 5, 10],
                  'penalty': ['l1', 'l2'],
                  'class_weight': ['unbalanced'],
                  'solver': ['saga'],
                  'tol': [0.01],
                  'max_iter': [1]},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 100,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500,
                   save_model = 'yes'):

    if use_gridsearch == 'yes':

        # Create model object
        model = GridSearchCV(LogisticRegression(), parameters, cv=3, verbose=10, scoring = 'roc_auc')

        # Fit the model
        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

        # View best hyperparameters
        print('Best Penalty:', model.best_estimator_.get_params()['penalty'])
        print('Best C:', model.best_estimator_.get_params()['C'])

        if refit == 'yes':

            # If refit, run
            model = LogisticRegression(C=model.best_estimator_.get_params()['C'],
                                       penalty=model.best_estimator_.get_params()['penalty'],
                                       solver=model.best_estimator_.get_params()['solver'],
                                       class_weight=model.best_estimator_.get_params()['class_weight'],
                                       max_iter=refit_iter,
                                       n_jobs=model.best_estimator_.get_params()['n_jobs'],
                                       tol=model.best_estimator_.get_params()['tol'],
                                       random_state=random_seed,
                                       verbose=10)

            # Refit
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = model.best_estimator_.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    elif use_saved_model == 'yes':

        # View best hyperparameters
        print('Saved Model Penalty:', saved_model.get_params()['penalty'])
        print('Saved Model C:', saved_model.get_params()['C'])

        if refit == 'yes':

            # If refit, run
            model = LogisticRegression(C=saved_model.get_params()['C'],
                                       penalty=saved_model.get_params()['penalty'],
                                       solver=saved_model.get_params()['solver'],
                                       class_weight=saved_model.get_params()['class_weight'],
                                       max_iter=refit_iter,
                                       n_jobs=saved_model.get_params()['n_jobs'],
                                       tol=saved_model.get_params()['tol'],
                                       random_state=random_seed,
                                       verbose=10)

            # Fit the model
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
    else:

        # Fit the model
        model = LogisticRegression(C=parameters['C'], penalty=parameters['penalty'], solver='saga',
                                   class_weight = parameters['class_weight'],
                                   max_iter = parameters['max_iter'],
                                   n_jobs = parameters['n_jobs'],
                                   tol=parameters['tol'],
                                   random_state = random_seed,
                                   verbose=10)

        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
        prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for Logistic Model"% (roc_auc_score(validation['click'], prediction[:, 1])))

    # Whether to save the model
    if save_model == 'yes':



    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]


# model, prediction = logistic_model(train2, validation1,
#                    parameters = {'C': 0.1, 'penalty': 'l1', 'class_weight': 'unbalanced', 'solver': 'saga',
#                                  'tol': 0.01, 'max_iter': 5,'n_jobs': 3},
#                    use_gridsearch = 'no',
#                    refit = 'no',
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = logistic_model(train2, validation1,
#                    parameters = {'C': [1, 2],
#                   'penalty': ['l1', 'l2'],
#                   'class_weight': ['unbalanced'],
#                   'solver': ['saga'],
#                   'tol': [0.01],
#                   'max_iter': [3],
#                                  'n_jobs': [3]},
#                    use_gridsearch = 'yes',
#                    refit = 'yes',
#                    refit_iter = 100,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
#
# model, prediction = logistic_model(train2, validation1,
#                    parameters = {'C': [1, 2],
#                   'penalty': ['l1', 'l2'],
#                   'class_weight': ['unbalanced'],
#                   'solver': ['saga'],
#                   'tol': [0.01],
#                   'max_iter': [3],
#                                  'n_jobs': [3]},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 13,
#                    use_saved_model = 'yes',
#                    saved_model = model,
#                    to_plot ='yes',
#                    random_seed = 500)


# --- RANDOM FOREST
def random_forest(train, validation,
                   parameters = {'max_depth': [2,3,4,5,6,7,8,9,10,11,12, None],
              'min_samples_split' :[4,5,6],
              "n_estimators" : [10],
              "min_samples_leaf": [1,2,3,4,5],
              "max_features": [4,5,6,"sqrt"],
              "criterion": ['gini','entropy']},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 100,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500):


    if use_gridsearch == 'yes':

        # Create model object
        model = GridSearchCV(RandomForestClassifier(), parameters, cv=3, verbose=10, scoring = 'roc_auc')

        # Fit the model
        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

        # View best hyperparameters
        print('Best Max Depth:', model.best_estimator_.get_params()['max_depth'])
        print('Best Min Sample Split:', model.best_estimator_.get_params()['min_samples_split'])
        print('Best Min Samples Leaf:', model.best_estimator_.get_params()['min_samples_leaf'])
        print('Best Max Features:', model.best_estimator_.get_params()['max_features'])
        print('Best Criterion:', model.best_estimator_.get_params()['criterion'])


        if refit == 'yes':

            # If refit, run
            model = RandomForestClassifier(max_depth=model.best_estimator_.get_params()["max_depth"]
                                   , max_features=model.best_estimator_.get_params()['max_features']
                                   , min_samples_leaf=model.best_estimator_.get_params()['min_samples_leaf']
                                   , min_samples_split=model.best_estimator_.get_params()['min_samples_split']
                                   , criterion=model.best_estimator_.get_params()['criterion']
                                   , n_estimators=refit_iter
                                   , n_jobs=3
                                   , verbose=10)

            # Refit
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = model.best_estimator_.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    elif use_saved_model == 'yes':

        # View saved model hyperparameters
        print('Saved Model Max Depth:', saved_model.get_params()['max_depth'])
        print('Saved Model Min Sample Split:', saved_model.get_params()['min_samples_split'])
        print('Saved Model Min Samples Leaf:', saved_model.get_params()['min_samples_leaf'])
        print('Saved Model Max Features:', saved_model.get_params()['max_features'])
        print('Saved Model Criterion:', saved_model.get_params()['criterion'])

        if refit == 'yes':

            # If refit, run
            model = RandomForestClassifier(max_depth=saved_model.get_params()["max_depth"]
                                   , max_features=saved_model.get_params()['max_features']
                                   , min_samples_leaf=saved_model.get_params()['min_samples_leaf']
                                   , min_samples_split=saved_model.get_params()['min_samples_split']
                                   , criterion=saved_model.get_params()['criterion']
                                   , n_estimators=refit_iter
                                   , n_jobs=3
                                   , verbose=10)
            # Fit the model
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
    else:

        # Fit the model
        model = RandomForestClassifier(max_depth=parameters["max_depth"]
                                       , max_features=parameters['max_features']
                                       , min_samples_leaf=parameters['min_samples_leaf']
                                       , min_samples_split=parameters['min_samples_split']
                                       , criterion=parameters['criterion']
                                       , n_estimators=parameters['n_estimators']
                                       , n_jobs=3
                                       , verbose=10)

        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
        prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for Random Forest Model"% (roc_auc_score(validation['click'], prediction[:, 1])))

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]

# model2, prediction = random_forest(train2, validation1,
#                    parameters = {'max_depth': [10,11,12, None],
#               'min_samples_split' :[4],
#               "n_estimators" : [10],
#               "min_samples_leaf": [1,2,3],
#               "max_features": ["sqrt"],
#               "criterion": ['entropy']},
#                    use_gridsearch = 'yes',
#                    refit = 'yes',
#                    refit_iter = 100,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = random_forest(train2, validation1,
#                    parameters = {'max_depth': None,
#               'min_samples_split' :4,
#               "n_estimators" : 10,
#               "min_samples_leaf": 2,
#               "max_features": "sqrt",
#               "criterion": 'entropy'},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 10,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = random_forest(train2, validation1,
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 500,
#                    use_saved_model = 'yes',
#                    saved_model = model2,
#                    to_plot ='yes',
#                    random_seed = 500)


# --- EXTREME RANDOM FOREST
def extreme_random_forest(train, validation,
                   parameters = {'max_depth': [2,3,4,5,6,7,8,9,10,11,12, None],
              'min_samples_split' :[4,5,6],
              "n_estimators" : [10],
              "min_samples_leaf": [1,2,3,4,5],
              "max_features": [4,5,6,"sqrt"],
              "criterion": ['gini','entropy']},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 100,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500):


    if use_gridsearch == 'yes':

        # Create model object
        model = GridSearchCV(ExtraTreesClassifier(), parameters, cv=3, verbose=10, scoring = 'roc_auc')

        # Fit the model
        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

        # View best hyperparameters
        print('Best Max Depth:', model.best_estimator_.get_params()['max_depth'])
        print('Best Min Sample Split:', model.best_estimator_.get_params()['min_samples_split'])
        print('Best Min Samples Leaf:', model.best_estimator_.get_params()['min_samples_leaf'])
        print('Best Max Features:', model.best_estimator_.get_params()['max_features'])
        print('Best Criterion:', model.best_estimator_.get_params()['criterion'])


        if refit == 'yes':

            # If refit, run
            model = ExtraTreesClassifier(max_depth=model.best_estimator_.get_params()["max_depth"]
                                   , max_features=model.best_estimator_.get_params()['max_features']
                                   , min_samples_leaf=model.best_estimator_.get_params()['min_samples_leaf']
                                   , min_samples_split=model.best_estimator_.get_params()['min_samples_split']
                                   , criterion=model.best_estimator_.get_params()['criterion']
                                   , n_estimators=refit_iter
                                   , n_jobs=3
                                   , verbose=10)

            # Refit
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = model.best_estimator_.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    elif use_saved_model == 'yes':

        # View saved model hyperparameters
        print('Saved Model Max Depth:', saved_model.get_params()['max_depth'])
        print('Saved Model Min Sample Split:', saved_model.get_params()['min_samples_split'])
        print('Saved Model Min Samples Leaf:', saved_model.get_params()['min_samples_leaf'])
        print('Saved Model Max Features:', saved_model.get_params()['max_features'])
        print('Saved Model Criterion:', saved_model.get_params()['criterion'])

        if refit == 'yes':

            # If refit, run
            model = ExtraTreesClassifier(max_depth=saved_model.get_params()["max_depth"]
                                   , max_features=saved_model.get_params()['max_features']
                                   , min_samples_leaf=saved_model.get_params()['min_samples_leaf']
                                   , min_samples_split=saved_model.get_params()['min_samples_split']
                                   , criterion=saved_model.get_params()['criterion']
                                   , n_estimators=refit_iter
                                   , n_jobs=3
                                   , verbose=10)
            # Fit the model
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
    else:

        # Fit the model
        model = ExtraTreesClassifier(max_depth=parameters["max_depth"]
                                       , max_features=parameters['max_features']
                                       , min_samples_leaf=parameters['min_samples_leaf']
                                       , min_samples_split=parameters['min_samples_split']
                                       , criterion=parameters['criterion']
                                       , n_estimators=parameters['n_estimators']
                                       , n_jobs=3
                                       , verbose=10)

        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
        prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for Extreme Random Forest Model"% (roc_auc_score(validation['click'], prediction[:, 1])))

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]


# model, prediction = extreme_random_forest(train2, validation1,
#                    parameters = {'max_depth': [10,11,12, None],
#               'min_samples_split' :[4],
#               "n_estimators" : [10],
#               "min_samples_leaf": [1,2,3],
#               "max_features": ["sqrt"],
#               "criterion": ['entropy']},
#                    use_gridsearch = 'yes',
#                    refit = 'yes',
#                    refit_iter = 100,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = extreme_random_forest(train2, validation1,
#                    parameters = {'max_depth': None,
#               'min_samples_split' :4,
#               "n_estimators" : 10,
#               "min_samples_leaf": 2,
#               "max_features": "sqrt",
#               "criterion": 'entropy'},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 10,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = extreme_random_forest(train2, validation1,
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 100,
#                    use_saved_model = 'yes',
#                    saved_model = model2,
#                    to_plot ='yes',
#                    random_seed = 500)

# --- GRADIENT BOOSTED TREES (XGBOOST)
def gradient_boosted_trees(train, validation,
                           parameters={'max_depth': [15, 20],
                                       "n_estimators": [10],
                                       "learning_rate": [0.05, 0.1],
                                       "colsample_bytees": [0.5],
                                       "reg_alpha": [0.1],
                                       "reg_lambda": [0.1],
                                       "subsample": [1]},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 20,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500):


    if use_gridsearch == 'yes':

        # Create model object
        model = GridSearchCV(xgboost.XGBClassifier(), parameters, cv=3, verbose=10, scoring = 'roc_auc')

        # Fit the model
        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

        # View best hyperparameters
        print('Saved Model Max Depth:', model.best_estimator_.get_params()['max_depth'])
        print('Saved Model Learning Rate:', model.best_estimator_.get_params()['learning_rate'])
        print('Saved Model Sub Sample:', model.best_estimator_.get_params()['subsample'])
        print('Saved Model Colsample By Trees:', model.best_estimator_.get_params()['colsample_bytree'])
        print('Saved Model Reg Alpha:', model.best_estimator_.get_params()['reg_alpha'])
        print('Saved Model Lambda:', model.best_estimator_.get_params()['reg_lambda'])


        if refit == 'yes':

            # If refit, run
            model = xgboost.XGBClassifier(max_depth=model.best_estimator_.get_params()['max_depth']
                                          , learning_rate=model.best_estimator_.get_params()['learning_rate']
                                          , subsample=model.best_estimator_.get_params()['subsample']
                                          , colsample_bytree=model.best_estimator_.get_params()['colsample_bytree']
                                          , reg_alpha=model.best_estimator_.get_params()['reg_alpha']
                                          , reg_lambda=model.best_estimator_.get_params()['reg_lambda']
                                          , n_estimators=refit_iter
                                          , n_jobs=3
                                          , verbose=10)

            # Refit
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = model.best_estimator_.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    elif use_saved_model == 'yes':

        # View saved model hyperparameters
        print('Saved Model Max Depth:', saved_model.get_params()['max_depth'])
        print('Saved Model Learning Rate:', saved_model.get_params()['learning_rate'])
        print('Saved Model Sub Sample:', saved_model.get_params()['subsample'])
        print('Saved Model Colsample By Trees:', saved_model.get_params()['colsample_bytree'])
        print('Saved Model Reg Alpha:', saved_model.get_params()['reg_alpha'])
        print('Saved Model Lambda:', saved_model.get_params()['reg_lambda'])

        if refit == 'yes':

            # If refit, run
            model = xgboost.XGBClassifier(max_depth=saved_model.get_params()['max_depth']
                                   , learning_rate=saved_model.get_params()['learning_rate']
                                   , subsample=saved_model.get_params()['subsample']
                                          , colsample_bytree=saved_model.get_params()['colsample_bytree']
                                          , reg_alpha=saved_model.get_params()['reg_alpha']
                                          , reg_lambda=saved_model.get_params()['reg_lambda']
                                   , n_estimators=refit_iter
                                   , n_jobs=3
                                   , verbose=10)
            # Fit the model
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
    else:

        # Fit the model
        model = xgboost.XGBClassifier(max_depth=parameters['max_depth']
                                      , learning_rate=parameters['learning_rate']
                                      , subsample=parameters['subsample']
                                      , colsample_bytree=parameters['colsample_bytree']
                                      , reg_alpha=parameters['reg_alpha']
                                      , reg_lambda=parameters['reg_lambda']
                                      , n_estimators=refit_iter
                                      , n_jobs=3
                                      , verbose=10)

        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
        prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for XGBoost Model"% (roc_auc_score(validation['click'], prediction[:, 1])))

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]


# model, prediction = gradient_boosted_trees(train2, validation1,
#                            parameters={'max_depth': [15, 20],
#                                        "n_estimators": [2],
#                                        "learning_rate": [0.05, 0.1],
#                                        "colsample_bytree": [0.5],
#                                        "reg_alpha": [0.1],
#                                        "reg_lambda": [0.1]},
#                    use_gridsearch = 'yes',
#                    refit = 'yes',
#                    refit_iter = 5,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = gradient_boosted_trees(train2, validation1,
#                            parameters={'max_depth': 20,
#                                        "n_estimators": 2,
#                                        "learning_rate": 0.1,
#                                        "colsample_bytree": 0.5,
#                                        "reg_alpha": 0.1,
#                                        "reg_lambda": 0.1,
#                                        "subsample": 1},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 5,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = gradient_boosted_trees(train2, validation1,
#                            parameters={'max_depth': 20,
#                                        "n_estimators": 2,
#                                        "learning_rate": 0.1,
#                                        "colsample_bytree": 0.5,
#                                        "reg_alpha": 0.1,
#                                        "reg_lambda": 0.1,
#                                        "subsample": 1},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 5,
#                    use_saved_model = 'yes',
#                    saved_model = model,
#                    to_plot ='yes',
#                    random_seed = 500)

# --- SUPPORT VECTOR MACHINES
def support_vector_machine(train, validation,
                           parameters={'C': [0.1, 1, 2],
                                       "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                                       "degree": [2, 3, 4],
                                       "gamma": ['auto'],
                                       "tol": [0.001],
                                       "max_iter": [10],
                                       "probability": True},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 20,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500):


    if use_gridsearch == 'yes':

        # Create model object
        model = GridSearchCV(svm.SVC(), parameters, cv=3, verbose=10, scoring = 'roc_auc')

        # Fit the model
        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

        # View best hyperparameters
        print('Saved Model C:', model.best_estimator_.get_params()['C'])
        print('Saved Kernel:', model.best_estimator_.get_params()['kernel'])

        if refit == 'yes':

            # If refit, run
            model = svm.SVC(C=model.best_estimator_.get_params()['C']
                                          , kernel=model.best_estimator_.get_params()['kernel']
                                          , degree=model.best_estimator_.get_params()['degree']
                                          , gamma=model.best_estimator_.get_params()['gamma']
                                          , tol=model.best_estimator_.get_params()['tol']
                                          , max_iter=refit_iter
                                          , verbose=10
                                , random_state = random_seed
                            ,probability=True)

            # Refit
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = model.best_estimator_.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    elif use_saved_model == 'yes':

        # View saved model hyperparameters
        print('Saved Model C:', saved_model.get_params()['C'])
        print('Saved Kernel:', saved_model.get_params()['kernel'])

        if refit == 'yes':

            # If refit, run
            model = svm.SVC(C=saved_model.get_params()['C']
                                          , kernel=saved_model.get_params()['kernel']
                                          , degree=saved_model.get_params()['degree']
                                          , gamma=saved_model.get_params()['gamma']
                                          , tol=saved_model.get_params()['tol']
                                          , max_iter=refit_iter
                                          , verbose=10
                            ,random_state = random_seed
                            ,probability=True)
            # Fit the model
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
    else:

        # Fit the model
        model = svm.SVC(C=parameters['C']
                                      , kernel=parameters['kernel']
                                      , degree=parameters['degree']
                                      , gamma=parameters['gamma']
                                      , tol=parameters['tol']
                                      , max_iter=refit_iter
                                      , verbose=10
                        , random_state=random_seed
                        , probability=True)

        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
        prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for SVM Model"% (roc_auc_score(validation['click'], prediction[:, 1])))

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]

# model, prediction = support_vector_machine(train2, validation1,
#                            parameters={'C': [0.1],
#                                        "kernel": ['poly', 'rbf'],
#                                        "degree": [3],
#                                        "gamma": ['auto'],
#                                        "tol": [0.001],
#                                        "max_iter": [10]},
#                    use_gridsearch = 'yes',
#                    refit = 'yes',
#                    refit_iter = 20,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = support_vector_machine(train2, validation1,
#                            parameters={'C': 0.1,
#                                        "kernel": 'rbf',
#                                        "degree": 3,
#                                        "gamma": 'auto',
#                                        "tol": 0.001,
#                                        "max_iter": 10,
#                                        'probability': True},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 20,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = support_vector_machine(train2, validation1,
#                            parameters={'C': 0.1,
#                                        "kernel": 'rbf',
#                                        "degree": 3,
#                                        "gamma": 'auto',
#                                        "tol": 0.001,
#                                        "max_iter": 10},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    refit_iter = 20,
#                    use_saved_model = 'yes',
#                    saved_model = model,
#                    to_plot ='yes',
#                    random_seed = 500)


# --- NAIVE BAYES
def naive_bayes(train, validation, to_plot ='yes'):

    # Fit the model
    model = GaussianNB()

    model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
    prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for Naive Bayes"% (roc_auc_score(validation['click'], prediction[:, 1])))

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]

naive_bayes(train2, validation1, to_plot ='yes')

# --- KNN
def KNN(train, validation,
                           parameters={'n_neighbors': [1, 2,3],
                                       "algorithm": ['auto']},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500):


    if use_gridsearch == 'yes':

        # Create model object
        model = GridSearchCV(KNeighborsClassifier(), parameters, cv=3, verbose=10, scoring = 'roc_auc')

        # Fit the model
        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

        # View best hyperparameters
        print('Best Model N Neighbours:', model.best_estimator_.get_params()['n_neighbors'])
        print('Best Algorithm:', model.best_estimator_.get_params()['algorithm'])

        if refit == 'yes':

            # If refit, run
            model = KNeighborsClassifier(n_neighbors=model.best_estimator_.get_params()['n_neighbors']
                                          , algorithm=model.best_estimator_.get_params()['algorithm'],
                                         n_jobs = 3
                                          , verbose=10)

            # Refit
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = model.best_estimator_.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    elif use_saved_model == 'yes':

        # View saved model hyperparameters
        print('Saved Model N Neighbours:', saved_model.get_params()['n_neighbors'])
        print('Saved Algorithm:', saved_model.get_params()['algorithm'])

        if refit == 'yes':

            # If refit, run
            model = KNeighborsClassifier(n_neighbors=saved_model.get_params()['n_neighbors']
                                         , algorithm=saved_model.get_params()['algorithm'],
                                         n_jobs=3
                                         , verbose=10)
            # Fit the model
            model = model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'])

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
    else:

        # Fit the model
        model = KNeighborsClassifier(n_neighbors=parameters['n_neighbors']
                                     , algorithm=parameters['algorithm'],
                                     n_jobs=3
                                     , verbose=10)


        model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])
        prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

    # Print scores
    print("AUC: %0.5f for KNN Model"% (roc_auc_score(validation['click'], prediction[:, 1])))

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]

# KNN(train2, validation1,
#                            parameters={'n_neighbors': [1, 2,3],
#                                        "algorithm": ['auto']},
#                    use_gridsearch = 'yes',
#                    refit = 'yes',
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = KNN(train, validation,
#                            parameters={'n_neighbors': 2,
#                                        "algorithm": 'auto'},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = KNN(train, validation,
#                            parameters={'n_neighbors': 2,
#                                        "algorithm": 'auto'},
#                    use_gridsearch = 'no',
#                    refit = 'yes',
#                    use_saved_model = 'yes',
#                    saved_model = model,
#                    to_plot ='yes',
#                    random_seed = 500)

# --- FIELD-AWARE FACTORIZATION MACHINE
def factorization_machine(train, validation,
                           parameters={'init_stdev': [0.1],
                                       "rank": [2, 3, 4],
                                       'l2_reg_w': [0.1, 1, 2],
                                       'l2_reg_V':[0.1, 0.5, 1],
                                       'n_iter': [10]},
                  # use_gridsearch = 'yes',
                   refit = 'yes',
                          refit_iter = 20,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500):

    # Transform the data to sparse representation
    train_X = train.drop(['click', 'bidprice', 'payprice'], axis=1)
    sparse_train_X = sp.csc_matrix(train_X)
    train_Y = train['click']
    train_Y[train_Y == 0] = -1

    validation_X = validation.drop(['click', 'bidprice', 'payprice'], axis=1)
    validation_Y = validation['click']
    validation_Y[validation_Y == 0] = -1
    sparse_validation_X = sp.csc_matrix(validation_X)

    # if use_gridsearch == 'yes':
    #
    #     # Create model object
    #     model = GridSearchCV(als.FMClassification(), parameters, cv=3, scoring = 'roc_auc')
    #
    #     # Fit the model
    #     model = model.fit(sparse_train_X, train_Y)
    #
    #     # View best hyperparameters
    #     print('Best Model Rank:', model.best_estimator_.get_params()['rank'])
    #     print('Best Model L2 Regularisation Parameter W:', model.best_estimator_.get_params()['l2_reg_w'])
    #     print('Best Model L2 Regularisation Parameter V:', model.best_estimator_.get_params()['l2_reg_V'])
    #
    #     if refit == 'yes':
    #
    #         # If refit, run
    #         model = als.FMClassification(rank=model.best_estimator_.get_params()['n_neighbors']
    #                                       , l2_reg_w=model.best_estimator_.get_params()['l2_reg_w']
    #                                      , l2_reg_V = model.best_estimator_.get_params()['l2_reg_V'],
    #                                      n_iter = refit_iter,
    #                                      random_state = random_seed)
    #
    #         # Refit
    #         model = model.fit(sparse_train_X, train_Y)
    #
    #         # Make prediction
    #         prediction = model.predict_proba(sparse_validation_X)
    #
    #     else:
    #         prediction = model.best_estimator_.predict_proba(sparse_validation_X)

    if use_saved_model == 'yes':

        # View saved model hyperparameters
        print('Saved Model Rank:', saved_model.get_params()['rank'])
        print('Saved Model L2 Regularisation Parameter W:', saved_model.get_params()['l2_reg_w'])
        print('Saved Model L2 Regularisation Parameter V:', saved_model.get_params()['l2_reg_V'])

        if refit == 'yes':

            # If refit, run
            model = als.FMClassification(rank=saved_model.get_params()['rank']
                                          , l2_reg_w=saved_model.get_params()['l2_reg_w']
                                         , l2_reg_V = saved_model.get_params()['l2_reg_V'],
                                         n_iter = refit_iter,
                                         random_state = random_seed)

            model = model.fit(sparse_train_X, train_Y)

            # Make prediction
            prediction = model.predict_proba(sparse_validation_X)

        else:
            prediction = saved_model.predict_proba(sparse_validation_X)
    else:

        # Fit the model
        model = als.FMClassification(rank=parameters['rank']
                                     , l2_reg_w=parameters['l2_reg_w']
                                     , l2_reg_V=parameters['l2_reg_V'],
                                     n_iter=parameters['n_iter'],
                                     random_state=random_seed)

        model = model.fit(sparse_train_X, train_Y)
        prediction = model.predict_proba(sparse_validation_X)

    # Print scores
    print("AUC: %0.5f for Factorization Machine Model"% (roc_auc_score(validation_Y, prediction)))

    if to_plot == 'yes':

        plot_ROC_curve(validation_Y, prediction)

    return model, prediction


# model, prediction = factorization_machine(train2, validation1,
#                            parameters={'init_stdev': 0.1,
#                                        "rank": 3,
#                                        'l2_reg_w': 1,
#                                        'l2_reg_V': 0.5,
#                                        'n_iter': 10},
#                  #  use_gridsearch = 'no',
#                    refit = 'yes',
#                           refit_iter = 20,
#                    use_saved_model = 'no',
#                    saved_model = [],
#                    to_plot ='yes',
#                    random_seed = 500)
#
# model, prediction = factorization_machine(train2, validation1,
#                            parameters={'init_stdev': 0.1,
#                                        "rank": 3,
#                                        'l2_reg_w': 1,
#                                        'l2_reg_V': 0.5,
#                                        'n_iter': 10},
#                  #  use_gridsearch = 'no',
#                    refit = 'yes',
#                           refit_iter = 20,
#                    use_saved_model = 'yes',
#                    saved_model = model,
#                    to_plot ='yes',
#                    random_seed = 500)


# --- STACKING
def stacking_classifier(train, validation, refit = 'yes', refit_iter = 20, use_saved_model = 'no',
                        saved_model = [], to_plot ='yes', random_seed = 500,
                        meta_model_parameters={'use_probas': False,
                                               'use_features_in_secondary': True,
                                               'cv': 5,
                                               'store_train_meta_features': True,
                                               'refit': True}):

    if use_saved_model == 'no':

        # Import best base models (from grid search done earlier)
        clf1 = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight='unbalanced', verbose=10,
                                  max_iter=10, n_jobs=3, tol=0.0001, random_state=random_state_nr)
        clf2 = LogisticRegression(C=1.3, penalty='l2', solver='saga', class_weight='unbalanced', verbose=10,
                                  max_iter=10, n_jobs=3, tol=0.0001, random_state=random_state_nr)
        clf3 = svm.SVC(C=1.0, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                       gamma='auto', kernel='rbf', max_iter=5, probability=True, random_state=None, shrinking=True,
                       tol=0.01, verbose=10)
        clf4 = GaussianNB()
        clf5 = RandomForestClassifier(n_estimators=50, max_depth=None,
                                      min_samples_split=5, min_samples_leaf=2,
                                      max_features='sqrt', criterion='entropy',
                                      verbose=10, n_jobs=3, random_state=random_state_nr)
        clf6 = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=5, random_state=random_state_nr,
                                    verbose=10, n_jobs=3, min_samples_leaf=2, max_features='sqrt',
                                    criterion='entropy', warm_start=True)
        clf7 = xgboost.XGBClassifier(max_depth=20, n_estimators=10, learning_rate=0.05, silent=False,
                                     n_jobs=3, subsample=1, objective='binary:logistic',
                                     colsample_bytree=1, eval_metric="auc", reg_alpha=0.1,
                                     reg_lambda=0.1)
        clf8 = xgboost.XGBClassifier(max_depth=20, n_estimators=20, learning_rate=0.05, silent=False,
                                     n_jobs=3, subsample=1, objective='binary:logistic',
                                     colsample_bytree=1, eval_metric="auc", reg_alpha=0.1,
                                     reg_lambda=0.1)
        sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6, clf7],
                                    meta_classifier=clf8, use_probas=meta_model_parameters['use_probas'],
                                    use_features_in_secondary = meta_model_parameters['use_features_in_secondary'],
                                    store_train_meta_features=meta_model_parameters['store_train_meta_features'],
                                    cv = meta_model_parameters['cv'])

        model = sclf.fit(train2.drop(['click', 'bidprice', 'payprice'], axis=1).values, train2['click'].values)
        prediction = model.predict_proba(validation1.drop(['click', 'bidprice', 'payprice'], axis=1).values)

    else:

        if refit == 'yes':

            # If refit, run
            model = saved_model.fit(train.drop(['click', 'bidprice', 'payprice'], axis=1).values, train['click'].values)

            # Make prediction
            prediction = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1).values)

        else:
            prediction = saved_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1).values)

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]


model, prediction = stacking_classifier(train2, validation1, refit_iter = 20, use_saved_model = 'no',
                        saved_model = [], to_plot ='yes', random_seed = 500,
                        meta_model_parameters={'use_probas': False,
                                               'use_features_in_secondary': True,
                                               'cv': 2,
                                               'store_train_meta_features': True,
                                               'refit': True})

model2, prediction2 = stacking_classifier(train2, validation1, refit = 'no', refit_iter = 20, use_saved_model = 'no',
                        saved_model = model, to_plot ='yes', random_seed = 500,
                        meta_model_parameters={'use_probas': False,
                                               'use_features_in_secondary': True,
                                               'cv': 2,
                                               'store_train_meta_features': True,
                                               'refit': True})







from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

import xgboost

random_state_nr = 500

clf1 = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight = 'unbalanced', verbose = 10,
                           max_iter = 100, n_jobs = 3, tol=0.0001, random_state = random_state_nr)
clf2 = LogisticRegression(C=1.3, penalty='l2', solver='saga', class_weight = 'unbalanced', verbose = 10,
                          max_iter = 100, n_jobs = 3, tol=0.0001, random_state = random_state_nr)
clf3 = svm.SVC(C=1.0, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
           gamma='auto', kernel='rbf', max_iter=5, probability=True, random_state=None, shrinking=True,
           tol=0.01, verbose=10)
# clf4 = KNeighborsClassifier(n_neighbors=2)
clf5 = GaussianNB()
clf6 = RandomForestClassifier(n_estimators=500, max_depth =None,
                                  min_samples_split = 5, min_samples_leaf = 2,
                                  max_features = 'sqrt', criterion ='entropy',
                                  verbose = 10, n_jobs = 3, random_state=random_state_nr)
clf7 = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=5, random_state=random_state_nr,
                                 verbose = 10, n_jobs = 3, min_samples_leaf = 2, max_features = 'sqrt',
                                 criterion='entropy', warm_start = True)
clf8 = xgboost.XGBClassifier(max_depth=20, n_estimators=100, learning_rate=0.05, silent = False,
                                  n_jobs = 3, subsample = 1, objective='binary:logistic',
                                  colsample_bytree = 1, eval_metric = "auc", reg_alpha = 0.1,
                                  reg_lambda = 0.1)
clf9 = xgboost.XGBClassifier(max_depth=20, n_estimators=200, learning_rate=0.05, silent = False,
                                  n_jobs = 3, subsample = 1, objective='binary:logistic',
                                  colsample_bytree = 1, eval_metric = "auc", reg_alpha = 0.1,
                                  reg_lambda = 0.1)
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf5, clf6, clf7, clf8],
                          meta_classifier=clf9, use_probas = False, use_features_in_secondary = True)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf5, clf6, clf7, clf8, sclf],
                      ['Logistic Regression (L1)',
                       'Logistic Regression (L2)',
                       'Support Vector Machines',
                      # 'KNN',
                       'Naive Bayes',
                       'Random Forest',
                       'Extreme Random Forest',
                       'XGBoost',
                       'StackingClassifier']):
    print(clf, label)

    scores = model_selection.cross_val_score(clf, train2.drop(['click','bidprice', 'payprice'], axis=1).values, train2['click'].values,
                                              cv=3, scoring='roc_auc')
    print("AUC: %0.5f (+/- %0.5f) [%s]"
          % (scores.mean(), scores.std(), label))
clf.predict_proba(validation1.drop(['click', 'bidprice', 'payprice'], axis=1))


clf1 = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight = 'unbalanced', verbose = 2,
                          max_iter = 100, n_jobs = 3, tol=0.0025, random_state = random_state_nr)
clf2 = LogisticRegression(C=1.3, penalty='l2', solver='saga', class_weight = 'unbalanced', verbose = 2,
                          max_iter = 100, n_jobs = 3, tol=0.0025, random_state = random_state_nr)
clf3 = RandomForestClassifier(n_estimators=100, max_depth =15,
                              min_samples_split = 5, min_samples_leaf = 2,
                              max_features = 'sqrt', criterion ='entropy',
                              verbose = 3, n_jobs = 3, random_state=random_state_nr)
clf4 = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=5, random_state=random_state_nr,
                            verbose = 2, n_jobs = 3, min_samples_leaf = 2, max_features = 'sqrt',
                            criterion='entropy', warm_start = True)
clf5 = xgboost.XGBClassifier(max_depth=20, n_estimators=30, learning_rate=0.05, silent = False,
                             n_jobs = 3, subsample = 0.5, objective='binary:logistic',
                             colsample_bytree=0.5, eval_metric = "auc", reg_alpha = 0.1,
                             reg_lambda = 0.1, random_state = random_state_nr)
lr = LogisticRegression()
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],
                          meta_classifier=lr, use_probas = True, use_features_in_secondary = True)

scores = model_selection.cross_val_score(sclf, train.drop(['click', 'bidprice', 'payprice'], axis=1), train['click'],
                                         cv=5, scoring='roc_auc')
print("AUC: %0.5f (+/- %0.5f) [%s]"
      % (scores.mean(), scores.std(), label))

# --- Factorization Machine
from fastFM import als
import scipy.sparse as sp

train_X = train2.drop(['click', 'bidprice', 'payprice'], axis = 1)
sparse_train_X = sp.csc_matrix(train_X)
train_Y = train2['click']
train_Y[train_Y==0] = -1

validation_X = validation1.drop(['click', 'bidprice', 'payprice'], axis = 1)
validation_Y = validation1['click']
validation_Y[validation_Y==0] = -1
sparse_validation_X = sp.csc_matrix(validation_X)

fm = als.FMClassification(n_iter=10, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
fm.fit(sparse_train_X, train_Y)


# Plot the errors
n_iter = 100
l2_reg_w = 0.1
l2_reg_V = 0.5
rank = 5
seed = 500
step_size = 1
values = np.arange(1, n_iter)

fm = als.FMClassification(n_iter=0, l2_reg_w=l2_reg_w,
                      l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
# Initalize coefs
fm.fit(sparse_train_X, train_Y)

roc_auc_train = []
roc_auc_validation = []
for i in range(1, n_iter):
    print(i)
    fm.fit(sparse_train_X, train_Y, n_more_iter=step_size)
    y_pred = fm.predict(sparse_validation_X)
    roc_auc_validation.append(roc_auc_score(validation_Y, fm.predict_proba(sparse_validation_X)))
    roc_auc_train.append(roc_auc_score(train_Y, fm.predict_proba(sparse_train_X)))

print('------- restart ----------')
roc_auc_validation_re = []
roc_auc_train_re = []
for i in values:
    print(i)
    fm = als.FMClassification(n_iter=i, l2_reg_w=l2_reg_w,
                              l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
    fm.fit(sparse_train_X, train_Y)
    roc_auc_validation_re.append(roc_auc_score(validation_Y, fm.predict_proba(sparse_validation_X)))
    roc_auc_train_re.append(roc_auc_score(train_Y, fm.predict_proba(sparse_train_X)))

from matplotlib import pyplot as plt

x = np.arange(1, n_iter) * step_size

with plt.style.context('fivethirtyeight'):
    plt.plot(x, roc_auc_train, label='train')
    plt.plot(x, roc_auc_validation, label='test')
    #plt.plot(values, roc_auc_validation_re, label='train re', linestyle='--')
   # plt.plot(values, roc_auc_train_re, label='test re', ls='--')
plt.legend()
plt.show()



# Prediction

y_pred = fm.predict(sparse_validation_X)
roc_auc_score(validation_Y, y_pred)
accuracy_score(validation_Y, y_pred)
confusion_matrix(validation_Y, fm.predict(sparse_validation_X))


fm.predict_proba(sparse_validation_X)



import numpy as np
import xlearn as xl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris_data = load_iris()
X = iris_data['data']
y = (iris_data['target'] == 2)

X_train,   \
X_val,     \
y_train,   \
y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# param:
#  0. binary classification
#  1. model scale: 0.1
#  2. epoch number: 10 (auto early-stop)
#  3. learning rate: 0.1
#  4. regular lambda: 1.0
#  5. use sgd optimization method
linear_model = xl.LRModel(task='binary', init=0.1,
                          epoch=10, lr=0.1,
                          reg_lambda=1.0, opt='sgd')

# Start to train
linear_model.fit(X_train, y_train,
                 eval_set=[X_val, y_val],
                 is_lock_free=False)

# Generate predictions
y_pred = linear_model.predict(X_val)



from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import numpy as np

RANDOM_SEED = 42

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression()

# The StackingCVClassifier uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN',
                       'Random Forest',
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X, y,
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


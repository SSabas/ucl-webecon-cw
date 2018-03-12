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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

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
                   random_seed = 500):

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

    if to_plot == 'yes':

        plot_ROC_curve(validation['click'], prediction[:, 1])

    return model, prediction[:,1]


model, prediction = logistic_model(train2, validation1,
                   parameters = {'C': 0.1, 'penalty': 'l1', 'class_weight': 'unbalanced', 'solver': 'saga',
                                 'tol': 0.01, 'max_iter': 5,'n_jobs': 3},
                   use_gridsearch = 'no',
                   refit = 'no',
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500)

model, prediction = logistic_model(train2, validation1,
                   parameters = {'C': [1, 2],
                  'penalty': ['l1', 'l2'],
                  'class_weight': ['unbalanced'],
                  'solver': ['saga'],
                  'tol': [0.01],
                  'max_iter': [3],
                                 'n_jobs': [3]},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 100,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500)


model, prediction = logistic_model(train2, validation1,
                   parameters = {'C': [1, 2],
                  'penalty': ['l1', 'l2'],
                  'class_weight': ['unbalanced'],
                  'solver': ['saga'],
                  'tol': [0.01],
                  'max_iter': [3],
                                 'n_jobs': [3]},
                   use_gridsearch = 'no',
                   refit = 'yes',
                   refit_iter = 13,
                   use_saved_model = 'yes',
                   saved_model = model,
                   to_plot ='yes',
                   random_seed = 500)


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

model2, prediction = random_forest(train2, validation1,
                   parameters = {'max_depth': [10,11,12, None],
              'min_samples_split' :[4],
              "n_estimators" : [10],
              "min_samples_leaf": [1,2,3],
              "max_features": ["sqrt"],
              "criterion": ['entropy']},
                   use_gridsearch = 'yes',
                   refit = 'yes',
                   refit_iter = 100,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500)

model, prediction = random_forest(train2, validation1,
                   parameters = {'max_depth': None,
              'min_samples_split' :4,
              "n_estimators" : 10,
              "min_samples_leaf": 2,
              "max_features": "sqrt",
              "criterion": 'entropy'},
                   use_gridsearch = 'no',
                   refit = 'yes',
                   refit_iter = 10,
                   use_saved_model = 'no',
                   saved_model = [],
                   to_plot ='yes',
                   random_seed = 500)

model, prediction = random_forest(train2, validation1,
                   use_gridsearch = 'no',
                   refit = 'yes',
                   refit_iter = 10,
                   use_saved_model = 'yes',
                   saved_model = model2,
                   to_plot ='yes',
                   random_seed = 500)

# Decision trees


parameters = {'max_depth': [2,3,4,5,6,7,8,9,10,11,12, None],
              'min_samples_split' :[4,5,6],
              "n_estimators" : [10],
              "min_samples_leaf": [1,2,3,4,5],
              "max_features": [4,5,6,"sqrt"],
              "criterion": ['gini','entropy']}


rf_regr = RandomForestClassifier()
rf_model = GridSearchCV(rf_regr,parameters, n_jobs = 3, cv = 3)
rf_model = rf_model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

learned_parameters = model_fit.best_params_  # Extract best params

rfc = RandomForestClassifier(max_depth = learned_parameters["max_depth"]
                            ,max_features = learned_parameters['max_features']
                            ,min_samples_leaf = learned_parameters['min_samples_leaf']
                            ,min_samples_split = learned_parameters['min_samples_split']
                            ,criterion = learned_parameters['criterion']
                            ,n_estimators = 5000
                            ,n_jobs = 3)

rf_model = RandomForestClassifier(n_estimators=20, max_depth =10,
                                  min_samples_split = 5, min_samples_leaf = 3,
                                  max_features = 'sqrt', criterion ='entropy',
                                  verbose = True, n_jobs = 3)

rf_model = rf_model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

print(rf_model.score(train.drop(['click','bidprice', 'payprice'], axis=1), train['click']))


rf_predict = rf_model.predict(validation.drop(['click','bidprice', 'payprice'], axis=1))


prediction = rf_model.predict(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = rf_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction)

# Plot ROC curve
plot_ROC_curve(validation['click'], prediction_proba[:, 1])



# --- EXTREME RANDOM FOREST


erf_model = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0,
                                 verbose = 2, n_jobs = 3, min_samples_leaf = 3, max_features = 'sqrt',
                                 criterion='entropy', warm_start = True)


erf_model = erf_model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

print(erf_model.score(train.drop(['click','bidprice', 'payprice'], axis=1), train['click']))


erf_predict = erf_model.predict(validation.drop(['click','bidprice', 'payprice'], axis=1))


prediction = erf_model.predict(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = erf_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction)

# Plot ROC curve
plot_ROC_curve(validation['click'], prediction_proba[:, 1])


# --- GRADIENT BOOSTED TREES (XGBOOST)
import xgboost


parameters = list(eta = 0.005,
      max_depth = 15,
      subsample = 0.5,
      colsample_bytree = 0.5,
      seed = 1,
      eval_metric = "auc",
      objective = "binary:logistic")

xgb_model = xgboost.XGBClassifier(max_depth=20, n_estimators=200, learning_rate=0.05, silent = False,
                                  n_jobs = 3, subsample = 0.5, objective='binary:logistic',
                                  colsample_bytree=0.5, eval_metric = "auc", reg_alpha = 0.1,
                                  reg_lambda = 0.1)

xgb_model = xgb_model.fit(train2.drop(['click','bidprice', 'payprice'], axis=1), train2['click'])

print(xgb_model.score(train.drop(['click','bidprice', 'payprice'], axis=1), train['click']))


xgb_predict = xgb_model.predict(validation1.drop(['click','bidprice', 'payprice'], axis=1))


prediction = xgb_model.predict(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = xgb_model.predict_proba(validation1.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction)

# Plot ROC curve
plot_ROC_curve(validation['click'], prediction_proba[:, 1])


# --- FIELD-AWARE FACTORIZATION MACHINE


# --- STACKING
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn import svm

import xgboost

random_state_nr = 500

clf1 = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight = 'unbalanced', verbose = 0,
                           max_iter = 100, n_jobs = 3, tol=0.0001, random_state = random_state_nr)
clf2 = LogisticRegression(C=1.3, penalty='l2', solver='saga', class_weight = 'unbalanced', verbose = 0,
                          max_iter = 100, n_jobs = 3, tol=0.0001, random_state = random_state_nr)
clf3 = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
           gamma='auto', kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True,
           tol=0.001, verbose=False)
clf4 = KNeighborsClassifier(n_neighbors=2)
clf5 = GaussianNB()
clf6 = RandomForestClassifier(n_estimators=500, max_depth =None,
                                  min_samples_split = 5, min_samples_leaf = 2,
                                  max_features = 'sqrt', criterion ='entropy',
                                  verbose = False, n_jobs = 3, random_state=random_state_nr)
clf7 = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=5, random_state=random_state_nr,
                                 verbose = 0, n_jobs = 3, min_samples_leaf = 2, max_features = 'sqrt',
                                 criterion='entropy', warm_start = True)
clf8 = xgboost.XGBClassifier(max_depth=20, n_estimators=100, learning_rate=0.05, silent = True,
                                  n_jobs = 3, subsample = 1, objective='binary:logistic',
                                  colsample_bytree = 1, eval_metric = "auc", reg_alpha = 0.1,
                                  reg_lambda = 0.1)
clf9 = xgboost.XGBClassifier(max_depth=20, n_estimators=200, learning_rate=0.05, silent = True,
                                  n_jobs = 3, subsample = 1, objective='binary:logistic',
                                  colsample_bytree = 1, eval_metric = "auc", reg_alpha = 0.1,
                                  reg_lambda = 0.1)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8],
                          meta_classifier=clf9, use_probas = False, use_features_in_secondary = True)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, sclf],
                      ['Logistic Regression (L1)',
                       'Logistic Regression (L2)',
                       'Support Vector Machines',
                       'KNN',
                       'Naive Bayes',
                       'Random Forest',
                       'Extreme Random Forest',
                       'XGBoost',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train2.drop(['click','bidprice', 'payprice'], axis=1), train2['click'],
                                              cv=3, scoring='roc_auc')
    print("AUC: %0.5f (+/- %0.5f) [%s]"
          % (scores.mean(), scores.std(), label))



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
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],
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



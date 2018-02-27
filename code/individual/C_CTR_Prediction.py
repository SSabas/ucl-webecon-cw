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
from sklearn.ensemble import RandomForestClassifier



# --------------------------------- FITTING --------------------------------------- #

# --- LOGISTIC REGRESSION
def logistic_model(train, validation, test, )

# Logistics regression
logistic = LogisticRegression()

# Parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10],
              'penalty': ['l1','l2'],
              'class_weight': ['balanced', 'unbalanced'],
              'solver': ['saga'],
              'tol': [0.01],
              'max_iter': [2]}

# Specify two scoring functions
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# Create model object
model = GridSearchCV(LogisticRegression(), param_grid, cv=3, verbose=1,
                     n_jobs = 3, scoring = scoring, refit ='AUC')

# Fit the model
model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])


# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression(C=1.3, penalty='l1', solver='saga', class_weight = 'balanced', verbose = True,
                           max_iter = 10, n_jobs = 3, tol=0.001)

param_grid = {'C': [0.1, 1, 10],
              'penalty': ['l1','l2'],
              'class_weight': ['balanced', 'unbalanced']}

model = model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])


prediction = model.predict(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction)



# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(validation['click'], prediction)
fpr, tpr, thresholds = roc_curve(validation['click'], prediction_proba[:, 1])

roc_auc = roc_auc_score(validation['click'], prediction)
roc_auc = roc_auc_score(validation['click'], prediction_proba[:, 1])


# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")




# --- DECISION TREES AND RANDOM FOREST
# Decision trees
from sklearn.ensemble import RandomForestClassifier


parameters = {'max_depth': [2,3,4,5,6,7,8,9,10,11,12],
              'min_samples_split' :[4,5,6],
              "n_estimators" : [10],
              "min_samples_leaf": [3,4,5],
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

rf_model = RandomForestClassifier(n_estimators=10, oob_score = True, max_depth =10,
                                  min_samples_split = 5, min_samples_leaf = 3,
                                  max_features = 'sqrt', criterion ='entropy',
                                  verbose = True, n_jobs = 3)

rf_model = rf_model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

print(rf_model.score(train.drop(['click','bidprice', 'payprice'], axis=1), train['click']))


rf_predict = rf_model.predict(validation.drop(['click','bidprice', 'payprice'], axis=1))
clf.oob_score_


prediction = rf_model.predict(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = rf_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction)



# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(validation['click'], prediction)
fpr, tpr, thresholds = roc_curve(validation['click'], prediction_proba[:, 1])

roc_auc = roc_auc_score(validation['click'], prediction)
roc_auc = roc_auc_score(validation['click'], prediction_proba[:, 1])


# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")





# --- GRADIENT BOOSTED TREES (XGBOOST)
import xgboost

xgb_model = xgboost.XGBClassifier(max_depth=3, n_estimators=1, learning_rate=0.05, silent = False,
                                  n_jobs = 3)


xgb_model = xgb_model.fit(train.drop(['click','bidprice', 'payprice'], axis=1), train['click'])

print(xgb_model.score(train.drop(['click','bidprice', 'payprice'], axis=1), train['click']))


xgb_predict = xgb_model.predict(validation.drop(['click','bidprice', 'payprice'], axis=1))


prediction = xgb_model.predict(validation.drop(['click', 'bidprice', 'payprice'], axis=1))
prediction_proba = xgb_model.predict_proba(validation.drop(['click', 'bidprice', 'payprice'], axis=1))

accuracy_score(validation['click'], prediction)
confusion_matrix(validation['click'], prediction)
roc_auc_score(validation['click'], prediction)



# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(validation['click'], prediction)
fpr, tpr, thresholds = roc_curve(validation['click'], prediction_proba[:, 1])

roc_auc = roc_auc_score(validation['click'], prediction)
roc_auc = roc_auc_score(validation['click'], prediction_proba[:, 1])


# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# --- FIELD-AWARE FACTORIZATION MACHINE


# Naive Bayes


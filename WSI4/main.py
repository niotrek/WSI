from ucimlrepo import fetch_ucirepo 
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets
y = y.values.ravel()
  
clf_dtc = DecisionTreeClassifier(random_state=42)
clf_svm = svm.SVC(kernel='linear', C=1, random_state=42)

metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# cross validation
cv_results_dtc = cross_validate(clf_dtc, X, y, cv=5, scoring=metrics)
cv_results_svm = cross_validate(clf_svm, X, y, cv=5, scoring=metrics)

for metric in metrics:
    test_scores_dtc = cv_results_dtc[f'test_{metric}']
    test_scores_svm = cv_results_svm[f'test_{metric}']
    # print(f"Decision Tree Classifier for {metric}")
    # print(f"Test scores: {test_scores_dtc}")   
    # print(f"SVM Classifier for {metric}")
    # print(f"Test scores: {test_scores_svm}")

results = pd.DataFrame()

random_seeds = [5, 23, 42]
for random_seed in random_seeds:
    # SVM measurements
    # regularization parameter
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for c in c_values:
        clf_svm = svm.SVC(kernel='linear', C=c, random_state=random_seed)
        cv_results_svm = cross_validate(clf_svm, X, y, cv=5, scoring=metrics)
        for metric in metrics:
            test_scores_svm = cv_results_svm[f'test_{metric}']
            mean_score = np.mean(test_scores_svm)
            std_dev = np.std(test_scores_svm)
            result = pd.DataFrame({
                'random_state': random_seed,
                'metric': metric, 
                'model': 'SVM', 
                'parameter': 'C', 
                'value': c, 
                'mean_score': mean_score,
                'std_dev': std_dev
                }, index=[0])            
            results = pd.concat([results, result], ignore_index=True)
    functions = ['linear', 'poly', 'rbf', 'sigmoid']
    for function in functions:
        clf_svm = svm.SVC(kernel=function, C=1, random_state=random_seed)
        cv_results_svm = cross_validate(clf_svm, X, y, cv=5, scoring=metrics)
        for metric in metrics:
            test_scores_svm = cv_results_svm[f'test_{metric}']
            mean_score = np.mean(test_scores_svm)
            std_dev = np.std(test_scores_svm)
            result = pd.DataFrame({
                'random_state': random_seed,
                'metric': metric, 
                'model': 'SVM', 
                'parameter': 'kernel', 
                'value': function, 
                'mean_score': mean_score,
                'std_dev': std_dev
                }, index=[0])
            results = pd.concat([results, result], ignore_index=True)
    iterations = [100, 1000, 10000, 100_000]
    for iteration in iterations:
        clf_svm = svm.SVC(kernel='linear', C=1, max_iter=iteration, random_state=random_seed, tol=1e-16)
        cv_results_svm = cross_validate(clf_svm, X, y, cv=5, scoring=metrics)
        for metric in metrics:
            test_scores_svm = cv_results_svm[f'test_{metric}']
            mean_score = np.mean(test_scores_svm)
            std_dev = np.std(test_scores_svm)
            result = pd.DataFrame({
                'random_state': random_seed,
                'metric': metric, 
                'model': 'SVM', 
                'parameter': 'max_iter', 
                'value': iteration, 
                'mean_score': mean_score,
                'std_dev': std_dev
                }, index=[0])
            results = pd.concat([results, result], ignore_index=True)

# mean_values = results.sort_values(by=['metric','parameter', 'value'])
# mean_values['mean_score'].agg(['mean']).reset_index()
# mean_values['std_dev'].agg(['mean']).reset_index()
# mean_values.to_csv('svm_mean_values.csv', index=False)

results = results.sort_values(by=['metric','parameter', 'value'])
mean_values = results.groupby(['metric', 'parameter', 'value']).agg({'mean_score': 'mean', 'std_dev': 'mean'}).reset_index()
mean_values.to_csv('svm_mean_values.csv', index=False)

results.to_csv('svm_results.csv', index=False)
          
# Decision Tree Classifier
results = pd.DataFrame()

for random_seed in random_seeds:
    # max depth
    max_depths = [10, 100, 1000, 10_000]
    for depth in max_depths:
        clf_dtc = DecisionTreeClassifier(random_state=random_seed, max_depth=depth)
        cv_results_dtc = cross_validate(clf_dtc, X, y, cv=5, scoring=metrics)
        for metric in metrics:
            test_scores_dtc = cv_results_dtc[f'test_{metric}']
            mean_score = np.mean(test_scores_dtc)
            std_dev = np.std(test_scores_dtc)
            result = pd.DataFrame({
                'random_state': random_seed,
                'metric': metric, 
                'model': 'DecisionTree', 
                'parameter': 'max_depth', 
                'value': depth, 
                'mean_score': mean_score,
                'std_dev': std_dev
                }, index=[0])            
            results = pd.concat([results, result], ignore_index=True)
    # criterion
    criterions = ['gini', 'entropy', 'log_loss']
    for cr in criterions:
        clf_dtc = DecisionTreeClassifier(random_state=random_seed, criterion=cr)
        cv_results_dtc = cross_validate(clf_dtc, X, y, cv=5, scoring=metrics)
        for metric in metrics:
            test_scores_dtc = cv_results_dtc[f'test_{metric}']
            mean_score = np.mean(test_scores_dtc)
            std_dev = np.std(test_scores_dtc)
            result = pd.DataFrame({
                'random_state': random_seed,
                'metric': metric, 
                'model': 'DecisionTree', 
                'parameter': 'criterion', 
                'value': cr, 
                'mean_score': mean_score,
                'std_dev': std_dev
                }, index=[0])
            results = pd.concat([results, result], ignore_index=True)
    # splitter
    splitters = ['best', 'random']
    for split in splitters:
        clf_dtc = DecisionTreeClassifier(random_state=random_seed, splitter=split)
        cv_results_dtc = cross_validate(clf_dtc, X, y, cv=5, scoring=metrics)
        for metric in metrics:
            test_scores_dtc = cv_results_dtc[f'test_{metric}']
            mean_score = np.mean(test_scores_dtc)
            std_dev = np.std(test_scores_dtc)
            result = pd.DataFrame({
                'random_state': random_seed,
                'metric': metric, 
                'model': 'DecisionTree', 
                'parameter': 'splitter', 
                'value': split, 
                'mean_score': mean_score,
                'std_dev': std_dev
                }, index=[0])
            results = pd.concat([results, result], ignore_index=True)

results = results.sort_values(by=['metric','parameter', 'value'])
mean_values = results.groupby(['metric', 'parameter', 'value']).agg({'mean_score': 'mean', 'std_dev': 'mean'}).reset_index()
mean_values.to_csv('dtc_mean_values.csv', index=False)
results.to_csv('dtc_results.csv', index=False)

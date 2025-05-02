import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score


def fetch_dataset():
    return pd.read_csv('Dataset/training_and_validation.csv', header=None) #Not even a header, not useful info from columns
    
def cut_dataset(dataset, indices):
    return dataset.drop(indices, axis=1).T.reset_index(drop=True).T
    
    
def create_train_test_sets(dataset):
    train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
    
    index_last = dataset.columns[-1]
    X_train = train_set.drop(index_last, axis=1)
    y_train = train_set[index_last]
    X_test = test_set.drop(index_last, axis=1)
    y_test = test_set[index_last]
    
    return (X_train, y_train, X_test, y_test)


def power_transform(X):
    pt = PowerTransformer(standardize=True)
    pt.set_output(transform='pandas')
    pt.fit(X)
    return pt.transform(X).T.reset_index().T.drop('index')

"""
Performs a gridsearch and prints the time taken and how the best found
predictor performs on the test set as well as some additional information
and optionally the scores for the next best models if show_top > 1
"""
def timed_gridsearch(gscv, X_train, y_train, X_test, y_test, show_top=0):
    
    print("Doing a gridsearch on", gscv.estimator)
    print("The parameters are:")
    for param in gscv.param_grid.keys():
        print(f"{param:20} {gscv.param_grid[param]}")
    print()
    
    start = time.time()
    gscv.fit(X_train, y_train)
    end = time.time()
    
    print("It took", round(end - start, 1), "seconds to perform a grid search.")

    estimator = gscv.best_estimator_
    start = time.time()
    y_test_pred = estimator.predict(X_test)
    end = time.time()
    
    print("It took", end - start, "seconds to predict the test set.")

    print("\nThe best predictor uses:")
    print(gscv.best_params_)
    print("and has an accuracy score of:", round(gscv.best_score_, 5))
    
    n_correct = sum(y_test_pred == y_test)
    accuracy = n_correct/len(y_test)
    
    cm = confusion_matrix(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print("\nOn the test set it scores:")
    print(f"{'Accuracy':30}  {accuracy: .6f}")
    print(f"{'Precision':30}  {precision: .6f}")
    print(f"{'Recall':30}  {recall: .6f}")
    print(f"{'f1 score':30}  {f1: .6f}")
    print(f"{'Confusion matrix':30}  {cm[0]} \n{'':30}  {cm[1]}")
    
    if (show_top):
        zipped = zip(gscv.cv_results_['params'], gscv.cv_results_['mean_test_score'])
        zipped_list = list(sorted(zipped, key= lambda t: t[1], reverse=True))
        
        print(f"\n The top {show_top} performers were:")
        for t in zipped_list[:show_top]:
            print(f"{str(t[0]):50} with accuracy{t[1]: .7f}")
            
    print("\n")
    
    return accuracy
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import code
import visualization as vis

"""
Run this program to easily make use of the functions written and get all the data insights used in the report.
Uncomment any part of the code to perform the associated action, 
by default no time consuming actions run automatically so not much will happen if you don't.
"""
    
if __name__ == "__main__":
    #Fetch the dataset:
    dataset = code.fetch_dataset()
    
    #dataset.info()
    #print(dataset)
    
    #Creates Fig. 1 in the report
    #vis.visualize_correlation_matrix(dataset)
    
    #Cut all but one of the highly correlated columns:
    dataset = code.cut_dataset(dataset, range(1,26))
    
    #vis.visualize_correlation_matrix(dataset)
    #print(dataset)
    
    #See the distribution of the features of the dataset
    #vis.make_histograms(dataset)
    
    #See distribution of one of the features before and after using transformers:
    #vis.test_transformers(dataset, 4)
    
    #Get test sets from the data
    X_train, y_train, X_test, y_test = code.create_train_test_sets(dataset)
    
    # Did not stratisfy so lets just check that saturation class is balanced in both sets... looks good!
    #sum(train_set[-1][:] == 1)/len(train_set)
    #sum(test_set[-1][:] == 1)/len(test_set)
    
    #Processing with PowerTransformer
    X_train_proc = code.power_transform(X_train)
    X_test_proc = code.power_transform(X_test)
    
    #sets contains the processed versions of X_train and X_test
    #Keep in mind that X_train and X_test are unprocessed.
    sets = (X_train_proc, y_train, X_test_proc, y_test)
    
    
    
    #To save time I added the results of various gridsearches in the following lines of code as comments:
    #(computation times reflect the use of 'Processor 13th Gen Intel(R) Core(TM) i5-13400F 2.50 GHz')

    sgdc = SGDClassifier(random_state=42)
    sgdc_parameters = {
        'penalty': ['l2', 'l1'],
        'loss': ['hinge', 'log_loss', 'modified_huber'],
        'alpha': np.logspace(-4, 4, 10),
        'max_iter': [1000, 2000]
    }
    sgdc_gscv=GridSearchCV(sgdc, sgdc_parameters, cv=5)
    #code.timed_gridsearch(sgdc_gscv, *sets, 5) #Uncomment to run gridsearch
    
    """
    It took 20.4 seconds to perform a grid search.
    It took 0.002999544143676758 seconds to predict the test set.

    The best predictor uses:
    {'alpha': np.float64(0.000774263682681127), 'loss': 'hinge', 'max_iter': 1000, 'penalty': 'l2'}
    and has an accuracy score of: 0.9855

    On the test set it scores:
    Accuracy                         0.985167
    Precision                        0.994259
    Recall                           0.976127
    f1 score                         0.985110
    Confusion matrix                [2967   17]
                                    [  72 2944]

    The top 5 performers were:
    {'alpha': np.float64(0.000774263682681127), 'loss': 'hinge',    'max_iter': 1000, 'penalty': 'l2'} with accuracy 0.9855000
    {'alpha': np.float64(0.000774263682681127), 'loss': 'hinge',    'max_iter': 2000, 'penalty': 'l2'} with accuracy 0.9855000
    {'alpha': np.float64(0.0001),               'loss': 'log_loss', 'max_iter': 1000, 'penalty': 'l2'} with accuracy 0.9847857
    {'alpha': np.float64(0.0001),               'loss': 'log_loss', 'max_iter': 2000, 'penalty': 'l2'} with accuracy 0.9847857
    {'alpha': np.float64(0.005994842503189409), 'loss': 'hinge',    'max_iter': 1000, 'penalty': 'l1'} with accuracy 0.9846429
    """


    dtc = DecisionTreeClassifier(random_state=42)
    dtc_parameters = {
        'max_depth': [5, 7, 9, 10, 12, 15, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': [5, 7, 9, 12, 15]
    }
    dtc_gscv=GridSearchCV(dtc, dtc_parameters, cv=5)
    #code.timed_gridsearch(dtc_gscv, *sets, 10) #Uncomment to run gridsearch

    """
    It took 16.7 seconds to perform a grid search.
    It took 0.0030367374420166016 seconds to predict the test set.

    The best predictor uses:
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 12, 'min_samples_split': 10}
    and has an accuracy score of: 0.99757

    On the test set it scores:
    Accuracy                         0.987833
    Precision                        0.994622
    Recall                           0.981101
    f1 score                         0.987815
    Confusion matrix                [2968   16]
                                    [  57 2959]

     The top 10 performers were:
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 12, 'min_samples_split': 10} with accuracy 0.9975714
    {'criterion': 'entropy', 'max_depth': 9, 'max_features': 12, 'min_samples_split': 10} with accuracy 0.9975000
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 12, 'min_samples_split': 2} with accuracy 0.9975000
    {'criterion': 'gini', 'max_depth': 9, 'max_features': 12, 'min_samples_split': 2} with accuracy 0.9972857
    ...
    """
    

    rfc = RandomForestClassifier(random_state=42)
    rfc_parameters = { 
        'max_depth': [5, 7, 9, 10],
        'n_estimators': [50, 100, 200, 500],
        'max_features': [5, 7, 9, 12, 15],
    }
    rfc_gscv=GridSearchCV(rfc, rfc_parameters, cv=5)
    #code.timed_gridsearch(rfc_gscv, *sets, 12) #Uncomment to run gridsearch
    
    """
    It took 999.8 seconds to perform a grid search.
    It took 0.034000396728515625 seconds to predict the test set.

    The best predictor uses:
    {'max_depth': 9, 'max_features': 12, 'n_estimators': 200}
    and has an accuracy score of: 0.99986

    On the test set it scores:
    Accuracy                         0.993500
    Precision                        0.997660
    Recall                           0.989390
    f1 score                         0.993508
    Confusion matrix                [2977    7]
                                    [  32 2984]

     The top 12 performers were:
    {'max_depth': 9, 'max_features': 12, 'n_estimators': 200} with accuracy 0.9998571
    {'max_depth': 9, 'max_features': 12, 'n_estimators': 500} with accuracy 0.9998571
    {'max_depth': 10, 'max_features': 7, 'n_estimators': 50} with accuracy 0.9998571
    ...
    """
    
    knnc = KNeighborsClassifier()
    knnc_parameters = {
        'leaf_size': [10, 20, 30, 40],
        'p': [1, 2, 5, 10],
        'n_neighbors': [1, 2, 3, 4, 5, 8, 10],
        'weights': ['uniform', 'distance']
    }
    knnc_gscv=GridSearchCV(knnc, knnc_parameters, cv=5)
    #code.timed_gridsearch(knnc_gscv, *sets, 10) #Uncomment to run gridsearch
    
    """
    It took 1506.2 seconds to perform a grid search.
    It took 0.2030029296875 seconds to predict the test set.

    The best predictor uses:
    {'leaf_size': 10, 'n_neighbors': 4, 'p': 1, 'weights': 'distance'}
    and has an accuracy score of: 0.99321

    On the test set it scores:
    Accuracy                         0.995333
    Precision                        0.995358
    Recall                           0.995358
    f1 score                         0.995358
    Confusion matrix                [2970   14]
                                    [  14 3002]

     The top 10 performers were:
    {'leaf_size': 10, 'n_neighbors': 4, 'p': 1, 'weights': 'distance'} with accuracy 0.9932143
    {'leaf_size': 20, 'n_neighbors': 4, 'p': 1, 'weights': 'distance'} with accuracy 0.9932143
    {'leaf_size': 30, 'n_neighbors': 4, 'p': 1, 'weights': 'distance'} with accuracy 0.9932143
    ...
    """
    
    nnc = MLPClassifier(random_state=42, verbose=False)
    nnc_parameters = { 
        'activation':['logistic', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'max_iter': [400]
    }
    nnc_gscv=GridSearchCV(nnc, nnc_parameters, cv=5)
    #code.timed_gridsearch(nnc_gscv, *sets, 6) #Uncomment to run gridsearch
    
    """
    It took 185.5 seconds to perform a grid search.
    It took 0.005000114440917969 seconds to predict the test set.

    The best predictor uses:
    {'activation': 'relu', 'max_iter': 400, 'solver': 'adam'}
    and has an accuracy score of: 0.99807

    On the test set it scores:
    Accuracy                         0.997167
    Precision                        0.996688
    Recall                           0.997679
    f1 score                         0.997183
    Confusion matrix                [2974   10]
                                    [   7 3009]

     The top 6 performers were:
    {'activation': 'relu',     'max_iter': 400, 'solver': 'adam'}  with accuracy 0.9980714
    {'activation': 'logistic', 'max_iter': 400, 'solver': 'adam'}  with accuracy 0.9977143
    {'activation': 'relu',     'max_iter': 400, 'solver': 'lbfgs'} with accuracy 0.9967857
    {'activation': 'logistic', 'max_iter': 400, 'solver': 'lbfgs'} with accuracy 0.9952857
    {'activation': 'relu',     'max_iter': 400, 'solver': 'sgd'}   with accuracy 0.9851429
    {'activation': 'logistic', 'max_iter': 400, 'solver': 'sgd'}   with accuracy 0.9803571
    """
    
    
    
    
    
    """
    In the following code we run 'gridsearches' of the best estimators we found to test
    the time it takes to fit the data and to predict the test set for each model.
    They're not really gridsearches if we only use our best predictors parameters of course
    But it was convenient to make use of the code.timed_gridsearch function even if
    the gridsearch only consists of a single cross validation fitting/training
    """
    sgdc = SGDClassifier(random_state=42)
    sgdc_parameters = {
        'penalty': ['l2'],
        'loss': ['hinge'],
        'alpha': [np.float64(0.000774263682681127)],
        'max_iter': [1000]
    }
    sgdc_gscv=GridSearchCV(sgdc, sgdc_parameters, cv=5)
    #code.timed_gridsearch(sgdc_gscv, *sets) #Uncomment to run timing test
    
    """
    It took 0.1 seconds to perform a grid search.
    It took 0.003000020980834961 seconds to predict the test set.

    The best predictor has an accuracy score of: 0.9855
    On the test set it scores:
    Accuracy                         0.985167
    """


    dtc = DecisionTreeClassifier(random_state=42)
    dtc_parameters = {
        'max_depth': [10],
        'min_samples_split': [10],
        'criterion': ['entropy'],
        'max_features': [12]
    }
    dtc_gscv=GridSearchCV(dtc, dtc_parameters, cv=5)
    #code.timed_gridsearch(dtc_gscv, *sets) #Uncomment to run timing test

    """
    It took 0.2 seconds to perform a grid search.
    It took 0.0030014514923095703 seconds to predict the test set.

    The best predictor has an accuracy score of: 0.99757
    On the test set it scores:
    Accuracy                         0.987833
    """
    

    rfc = RandomForestClassifier(random_state=42)
    rfc_parameters = { 
        'max_depth': [9],
        'n_estimators': [200],
        'max_features': [12],
    }
    rfc_gscv=GridSearchCV(rfc, rfc_parameters, cv=5)
    #code.timed_gridsearch(rfc_gscv, *sets) #Uncomment to run timing test
    
    """
    It took 18.0 seconds to perform a grid search.
    It took 0.034000396728515625 seconds to predict the test set.

    The best predictor has an accuracy score of: 0.99986
    On the test set it scores:
    Accuracy                         0.993500
    """


    knnc = KNeighborsClassifier()
    knnc_parameters = {
        'leaf_size': [10],
        'p': [1],
        'n_neighbors': [4],
        'weights': ['distance']
    }
    knnc_gscv=GridSearchCV(knnc, knnc_parameters, cv=5)
    #code.timed_gridsearch(knnc_gscv, *sets) #Uncomment to run timing test
    
    """
    It took 0.6 seconds to perform a grid search.
    It took 0.18800044059753418 seconds to predict the test set.

    The best predictor has an accuracy score of: 0.99321
    On the test set it scores:
    Accuracy                         0.995333
    """
    
    nnc = MLPClassifier(random_state=42, verbose=False)
    nnc_parameters = { 
        'activation':['relu'],
        'solver': ['adam'],
        'max_iter': [400]
    }
    nnc_gscv=GridSearchCV(nnc, nnc_parameters, cv=5)
    #code.timed_gridsearch(nnc_gscv, *sets) #Uncomment to run timing test
    
    """
    It took 14.6 seconds to perform a grid search.
    It took 0.004290103912353516 seconds to predict the test set.

    The best predictor has an accuracy score of: 0.99807
    On the test set it scores:
    Accuracy                         0.997167
    """
    
    
    
    #Run this to create Fig. 2 in the report as well as see the accuracy/f1 scores for each test
    #vis.plot_performance_using_few_features(*sets, 12)
    
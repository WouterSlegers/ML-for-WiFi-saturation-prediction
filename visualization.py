import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler

import code
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

"""
Generates a histogram for each column of the dataset
"""
def make_histograms(dataset):
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    # Generate histograms for each numerical column in the housing DataFrame
    dataset.hist(bins=50, figsize=(25, 16)) # Plot histograms with 50 bins and a figure size of 12x8 inches
    #save_fig("attribute_histogram_plots")  # extra code
    plt.show()

"""
Shows some of the information that can be gained from the correlation matrix
in our dataset. Also returns the correlation matrix so it can be explored further.
"""    
def visualize_correlation_matrix(dataset):
    corr_matrix = dataset.corr()
    print('Correlation between first feature and all the others:')
    print(corr_matrix[0])
    
    index_last = dataset.columns[-1]
    plt.title('Absolute correlation feature to be predicted')
    plt.xlabel('Columns of the dataset')
    plt.ylabel('Correlation to the label that is to be predicted')
    plt.plot(abs(corr_matrix[index_last][:index_last]))
    plt.show()
    return corr_matrix
    
"""
Shows the distribution of a given column before and after using a PowerTransformer
and a FunctionTransformer using a logarithm function
"""
def test_transformers(dataset, x):
    lt = FunctionTransformer(np.log, inverse_func=np.exp)
    sc = StandardScaler()
    sc.set_output(transform='pandas')
    
    pt = PowerTransformer()
    pt.set_output(transform='pandas')


    lt.fit(dataset+0.01)
    ltd = lt.transform(dataset+0.01)
    sc.fit(ltd)
    ltd = sc.transform(ltd)

    pt.fit(dataset)
    ptd = pt.transform(dataset)

    dataset.hist(dataset.columns[x], bins=50, figsize=(10,7))
    plt.title('orig')
    ltd.hist(ltd.columns[x], bins=50, figsize=(10,7))
    plt.title('log')
    ptd.hist(ptd.columns[x], bins=50, figsize=(10,7))
    plt.title('power')

    plt.show()

"""
A travesty of a function, it makes Fig. 2 in the report and keeps you updated
each step of the way how the models perform on the test set after having used
only i features from the dataset during the learning and testing stages
"""
def plot_performance_using_few_features(X_train, y_train, X_test, y_test, max_features):
    
    rfc = RandomForestClassifier(random_state=42)
    rfc_parameters = { 
        'max_depth': [9],
        'n_estimators': [200],
        'max_features': [12],
    }
    rfc_gscv=GridSearchCV(rfc, rfc_parameters, cv=5)
    rfc_gscv.fit(X_train, y_train)
    feat_importance = rfc_gscv.best_estimator_.feature_importances_
    sorted_list = list(sorted(zip(range(0,29), feat_importance), key=lambda t:t[1], reverse=True))
    indices_sorted = [t[0] for t in sorted_list]
    print("Features ordered by importance given by RFc:\n", indices_sorted)

    indices_sorted = [28, 1, 0, 6, 7, 8, 27, 3, 14, 15, 2, 16, 17, 9, 18, 26, 5, 13, 11, 19, 10, 20, 12, 21, 25, 22, 4, 23, 24]
    ind, accuracies_rf, accuracies_nn = list(), list(), list()
    
    for i in range(1, max_features+1):
        ind.append(i)
        
        print("Using only the following", i, "features:")
        print(indices_sorted[:i])
        
        rfc = RandomForestClassifier(random_state=42)
        rfc_parameters = { 
            'max_depth': [9],
            'n_estimators': [200],
            'max_features': [12],
        }
        rfc_gscv=GridSearchCV(rfc, rfc_parameters, cv=5)
        accuracies_rf.append(code.timed_gridsearch(rfc_gscv, X_train[indices_sorted[:i]], y_train, X_test[indices_sorted[:i]], y_test))
        
        nnc = MLPClassifier(random_state=42, verbose=False)
        nnc_parameters = { 
            'activation':['relu'],
            'solver': ['adam'],
            'max_iter': [800]
        }
        nnc_gscv=GridSearchCV(nnc, nnc_parameters, cv=5)
        accuracies_nn.append(code.timed_gridsearch(nnc_gscv, X_train[indices_sorted[:i]], y_train, X_test[indices_sorted[:i]], y_test))
        
    
    print(accuracies_rf)
    print(accuracies_nn)
    plt.title("Accuracy on test set using fewer features")
    plt.xlabel("Features used")
    plt.ylabel("Accuracy on test set")
    plt.plot(ind, accuracies_rf, label='Random Forest')
    plt.plot(ind, accuracies_nn, label='Neural Network')
    plt.legend(['Random Forest', 'Neural Network'])
    plt.show()
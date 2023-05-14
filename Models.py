import nltk
import math
import scipy
import sklearn
import spacy
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, normalized_mutual_info_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

class Models:
    def getScores(self, y_test, y_pred):
        """
        Calculates various evaluation scores based on the predicted and true labels.

        Args:
            y_test (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            dict: A dictionary containing the evaluation scores.

        """
        accuracy = accuracy_score(y_test, y_pred)
        nmi = normalized_mutual_info_score(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm4 = self.convertToCM4(cm)
        precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        return  {'acc': accuracy, 'nmi': nmi, 'prec': precision,  'rec': recall, 'cm': cm, 'cm4': cm4,}

    def runRandomForestClassifier(self, X_train, y_train, X_test):
        """
        Runs a Random Forest Classifier on the given training features and predicts the labels for the testing features.

        Args:
            X_train (pandas.DataFrame): The training features.
            y_train (array-like): The training labels.
            X_test (pandas.DataFrame): The testing features.

        Returns:
            array-like: The predicted labels for the testing features.

        """
        rf = RandomForestClassifier(random_state=42, max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 200, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)   

        return y_pred
    
    def runLinearSVC(self, X_train, y_train, X_test):       
        """
        Runs a Linear Support Vector Classifier (SVC) on the given training features and predicts the labels for the testing features.

        Args:
            X_train (pandas.DataFrame): The training features.
            y_train (array-like): The training labels.
            X_test (pandas.DataFrame): The testing features.

        Returns:
            array-like: The predicted labels for the testing features.

        """ 
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test) 

        return y_pred
    
    def runXGBoost(self, X_train, y_train, X_test):     
        """
        Runs XGBoost Classifier on the given training features and predicts the labels for the testing features.

        Args:
            X_train (pandas.DataFrame): The training features.
            y_train (array-like): The training labels.
            X_test (pandas.DataFrame): The testing features.

        Returns:
            array-like: The predicted labels for the testing features.

        """
        xgb_model = xgb.XGBClassifier(
                                        max_depth=3,
                                        learning_rate=0.1,
                                        n_estimators=100,
                                        verbosity=1,
                                        objective='binary:logistic',
                                        booster='gbtree',
                                        random_state=42,
                                        n_jobs = -1
                                     )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        return y_pred
    
    def runKMeans(self, X, clusters):     
        """
        Runs K-Means clustering on the given features and predicts the cluster labels.

        Args:
            X (pandas.DataFrame): The features.
            clusters (int): The number of clusters.

        Returns:
            array-like: The predicted cluster labels.

        """
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        y_pred = kmeans.fit_predict(X)

        return y_pred
    
    def runGaussianMixture(self, X, components):   
        """
        Runs Gaussian Mixture Model clustering on the given features and predicts the cluster labels.

        Args:
            X (pandas.DataFrame): The features.
            components (int): The number of components/clusters.

        Returns:
            array-like: The predicted cluster labels.

        """  
        gmm = GaussianMixture(n_components=components, random_state=42)
        y_pred = gmm.fit_predict(X)

        return y_pred
    
    def runMLP(self, X_train, y_train, X_test):        
        """
        Runs a Multilayer Perceptron (MLP) Classifier on the given training features and predicts the labels for the testing features.

        Args:
            X_train (pandas.DataFrame): The training features.
            y_train (array-like): The training labels.
            X_test (pandas.DataFrame): The testing features.

        Returns:
            array-like: The predicted labels for the testing features.

        """
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
        # Fit the MLP classifier to the training data
        mlp.fit(X_train, y_train)

        # Predict labels on the test data
        y_pred = mlp.predict(X_test)

        return y_pred
    
    def convertToCM4(self, cm):
        """
        Converts the given confusion matrix to a 2x2 matrix of True Positive, False Positive, False Negative, and True Negative.

        Args:
            cm (array-like): The confusion matrix.

        Returns:
            numpy.ndarray: The converted 2x2 matrix.

        """
        n = len(cm)

        # Convert to nxn Matrix of Total True/False Positive and True/False Negative
        tp = [cm[i][i] for i in range(n)]
        fp = [sum([cm[j][i] for j in range(n)]) - cm[i][i] for i in range(n)]
        fn = [sum([cm[i][j] for j in range(n)]) - cm[i][i] for i in range(n)]
        tn = sum([sum(cm[i]) for i in range(n)]) - sum(tp) - sum(fp) - sum(fn)

        tp_nxn = sum(tp)
        fp_nxn = sum(fp)
        fn_nxn = sum(fn)
        tn_nxn = tn

        matrix_nxn = [[tp_nxn, fp_nxn], [fn_nxn, tn_nxn]]
        
        return np.array(matrix_nxn)

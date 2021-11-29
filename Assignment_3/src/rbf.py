#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:04 2021

IMC: lab assignment 3

@author: pagutierrez
@author: Ventura Lucena Martínez
    References: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
"""

import pickle
import os
import click
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import squareform
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import time


@click.command()
@click.option('--train_file', '-t', default=None, required=True,
              help=u'Name of the file with training data.')
@click.option('--test_file', '-T', default=None, required=False,
              help=u'Name of the file with test data.')
@click.option('--classification', '-c', is_flag=True, default=False, show_default=True, required=False,
              help=u'Boolean that indicates wether it is a classification problem or not.')
@click.option('--ratio_rbf', '-r', default=0.1, show_default=True, required=False,
              help=u'Indicates the radius (by one) of RBF neurons with respect to the total number of patterns in training.')
@click.option('--l2', '-l', is_flag=True, default=False, show_default=True, required=False,
              help=u'Boolean that indicates if L2 regularization is used. If it is not specified, L1 will be used.')
@click.option('--eta', '-e', default=1e-2, show_default=True, required=False,
              help=u'Indicates the value for eta (η) parameter.')
@click.option('--outputs', '-o', default=1, show_default=True, required=False,
              help=u'Indicates the number of output columns of the dataset (always placed at the end).')
@click.option('--pred', '-p', is_flag=True, default=False, show_default=True,
              help=u'Use the prediction mode.')  # KAGGLE
@click.option('--model', '-m', default="", show_default=False,
              help=u'Directory name to save the models (or name of the file to load the model, if the prediction mode is active).')  # KAGGLE
def train_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model, pred):
    """ 5 executions of RBFNN training

        RBF neural network based on hybrid supervised/unsupervised training.
        We run 5 executions with different seeds.
    """
    start = time.time()

    if not pred:

        if train_file is None:
            print("You have not specified the training file (-t)")
            return

        if test_file is None:
            test_file = train_file

        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)

        for s in range(1, 6, 1):
            print("-----------")
            print("Seed: %d" % s)
            print("-----------")
            np.random.seed(s)
            train_mses[s-1], test_mses[s-1], train_ccrs[s-1], test_ccrs[s-1] = \
                train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs,
                          model and "{}/{}.pickle".format(model, s) or "")
            print("Training MSE: %f" % train_mses[s-1])
            print("Test MSE: %f" % test_mses[s-1])
            print("Training CCR: %.2f%%" % train_ccrs[s-1])
            print("Test CCR: %.2f%%" % test_ccrs[s-1])

        print("******************")
        print("Summary of results")
        print("******************")
        print("Training MSE: %f +- %f" %
              (np.mean(train_mses), np.std(train_mses)))
        print("Test MSE: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("Training CCR: %.2f%% +- %.2f%%" %
              (np.mean(train_ccrs), np.std(train_ccrs)))
        print("Test CCR: %.2f%% +- %.2f%%" %
              (np.mean(test_ccrs), np.std(test_ccrs)))

    else:
        # KAGGLE
        if model is None:
            print("You have not specified the file with the model (-m).")
            return

        # Obtain the predictions for the test set
        predictions = predict(test_file, model)

        # Print the predictions in csv format
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""
            s += str(index)

            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))

            print(s)

    end = time.time()
    print("\nTime (s) = ", end - start)


def train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ One execution of RBFNN training

        RBF neural network based on hybrid supervised/unsupervised training.
        We run 1 executions.

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        outputs: int
            Number of variables that will be used as outputs (all at the end
            of the matrix)
        model_file: string
            Name of the file where the model will be written

        Returns
        -------
        train_mse: float
            Mean Squared Error for training data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        test_mse: float 
            Mean Squared Error for test data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        train_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
        test_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
    """
    train_inputs, train_outputs, test_inputs, test_outputs = read_data(train_file,
                                                                       test_file,
                                                                       outputs)

    # [x] TODO: Obtain num_rbf from ratio_rbf
    num_rbf = int(ratio_rbf * len(train_inputs))

    print("Number of RBFs used: %d" % (num_rbf))
    kmeans, distances, centers = clustering(classification, train_inputs,
                                            train_outputs, num_rbf)

    radii = calculate_radii(centers, num_rbf)

    r_matrix = calculate_r_matrix(distances, radii)

    if not classification:
        coefficients = invert_matrix_regression(r_matrix, train_outputs)
    else:
        logreg = logreg_classification(r_matrix, train_outputs, l2, eta)

    """
    [x] TODO: Obtain the distances from the centroids to the test patterns
          and obtain the R matrix for the test set
    """
    r_matrix_test = calculate_r_matrix(kmeans.transform(test_inputs), radii)

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification': classification,
            'radii': radii,
            'kmeans': kmeans
        }
        if not classification:
            save_obj['coefficients'] = coefficients
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)

    # # # # # # # # # # #

    if not classification:
        """
        [x] TODO: Obtain the predictions for training and test and calculate
              the MSE
        """
        train_prediction = np.matmul(r_matrix, coefficients)
        test_prediction = np.matmul(r_matrix_test, coefficients)

        train_mse = mean_squared_error(train_prediction, train_outputs)
        test_mse = mean_squared_error(test_prediction, test_outputs)

        train_ccr = 0
        test_ccr = 0
        # Experiment 4:
        #train_ccr = int(np.mean(train_prediction))
        #test_ccr = int(np.mean(test_prediction))
    else:
        """
        [x] TODO: Obtain the predictions for training and test and calculate
              the CCR. Obtain also the MSE, but comparing the obtained
              probabilities and the target probabilities
        """
        test_predictions = logreg.predict(r_matrix_test)

        tr_predictions = logreg.predict_proba(r_matrix)
        te_predictions = logreg.predict_proba(r_matrix_test)

        train_mse = mean_squared_error(
            tr_predictions, OneHotEncoder().fit_transform(train_outputs).toarray())
        test_mse = mean_squared_error(
            te_predictions, OneHotEncoder().fit_transform(test_outputs).toarray())

        train_ccr = logreg.score(r_matrix, train_outputs) * 100
        test_ccr = logreg.score(r_matrix_test, test_outputs) * 100

        print("\nConfusion matrix")
        print(confusion_matrix(test_outputs, test_predictions))
        print()

    return train_mse, test_mse, train_ccr, test_ccr


def read_data(train_file, test_file, outputs):
    """ Read the input data
        It receives the name of the input data file names (training and test)
        and returns the corresponding matrices

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        outputs: int
            Number of variables to be used as outputs
            (all at the end of the matrix).

        Returns
        -------
        train_inputs: array, shape (n_train_patterns,n_inputs)
            Matrix containing the inputs for the training patterns
        train_outputs: array, shape (n_train_patterns,n_outputs)
            Matrix containing the outputs for the training patterns
        test_inputs: array, shape (n_test_patterns,n_inputs)
            Matrix containing the inputs for the test patterns
        test_outputs: array, shape (n_test_patterns,n_outputs)
            Matrix containing the outputs for the test patterns
    """

    # [x] TODO: Complete the code of the function
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()

    train_inputs = train_np[:, 0:-outputs]
    train_outputs = train_np[:, train_inputs.shape[1]:]

    test_inputs = test_np[:, 0:-outputs]
    test_outputs = test_np[:, test_inputs.shape[1]:]

    return train_inputs, train_outputs, test_inputs, test_outputs


def init_centroids_classification(train_inputs, train_outputs, num_rbf):
    """ Initialize the centroids for the case of classification
        This method selects in a stratified num_rbf patterns.

        Parameters
        ----------
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network

        Returns
        -------
        centroids: array, shape (num_rbf,n_inputs)
            Matrix with all the centroids already selected
    """

    # [x] TODO: Complete the code of the function
    x_train, x_test, y_train, y_test = train_test_split(
        train_inputs, train_outputs, train_size=num_rbf, stratify=train_outputs)

    centroids = x_train
    return centroids


def clustering(classification, train_inputs, train_outputs, num_rbf):
    """ It applies the clustering process
        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification(). 
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network

        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        centers: array, shape (num_rbf,n_inputs)
            Centers after the clustering
    """

    # [x] TODO: Complete the code of the function
    if classification:
        centroids = init_centroids_classification(
            train_inputs, train_outputs, num_rbf)
        kmeans = KMeans(n_clusters=num_rbf, init=centroids,
                        n_init=1, max_iter=500)
    else:
        kmeans = KMeans(n_clusters=num_rbf, init='random',
                        n_init=1, max_iter=500)

    kmeans.fit(train_inputs)

    distances = kmeans.transform(train_inputs)
    centers = kmeans.cluster_centers_

    return kmeans, distances, centers


def calculate_radii(centers, num_rbf):
    """ It obtains the value of the radii after clustering
        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Parameters
        ----------
        centers: array, shape (num_rbf,n_inputs)
            Centers from which obtain the radii
        num_rbf: int
            Number of RBFs to be used in the network

        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
    """

    # [x] TODO: Complete the code of the function
    radii = np.array([])
    distances = squareform(pdist(centers))

    for i in range(num_rbf):
        radii = np.append(radii, sum(distances[i] / (2 * (num_rbf - 1))))

    return radii


def calculate_r_matrix(distances, radii):
    """ It obtains the R matrix
        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias

        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF

        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
    """

    # [x] TODO: Complete the code of the function
    net = np.power(distances, 2)

    for i in range(net.shape[1]):
        net[:, i] = np.exp(-net[:, i] / (2 * np.power(radii[i], 2)))

    out = net
    ones = np.ones((out.shape[0], 1))
    r_matrix = np.hstack((out, ones))

    return r_matrix


def invert_matrix_regression(r_matrix, train_outputs):
    """ Inversion of the matrix for regression case
        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression

        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value 
            of the bias 
    """

    # [x] TODO: Complete the code of the function
    pseudo_inverse = np.linalg.pinv(r_matrix)
    coefficients = np.matmul(pseudo_inverse, train_outputs)

    return coefficients


def logreg_classification(r_matrix, train_outputs, l2, eta):
    """ Performs logistic regression training for the classification case
        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)

        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression

        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression
            Scikit-learn logistic regression model already trained
    """

    # [x] TODO: Complete the code of the function
    if l2:
        logreg = LogisticRegression(C=(1 / eta), solver='liblinear')
    else:
        logreg = LogisticRegression(
            penalty='l1', C=(1 / eta), solver='liblinear')

    logreg.fit(r_matrix, train_outputs.ravel())

    return logreg


def predict(test_file, model_file):
    """ Performs a prediction with RBFNN model
        It obtains the predictions of a RBFNN model for a test file, using two files, one
        with the test data and one with the model

        Parameters
        ----------
        test_file: string
            Name of the test file
        model_file: string
            Name of the file containing the model data

        Returns
        -------
        test_predictions: array, shape (n_test_patterns,n_outputs)
            Predictions obtained with the model and the test file inputs
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]

    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)

    radii = saved_data['radii']
    classification = saved_data['classification']
    kmeans = saved_data['kmeans']

    test_distancias = kmeans.transform(test_inputs)
    test_r = calculate_r_matrix(test_distancias, radii)

    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coefficients']
        test_predictions = np.dot(test_r, coeficientes)

    return test_predictions


if __name__ == "__main__":
    train_rbf_total()

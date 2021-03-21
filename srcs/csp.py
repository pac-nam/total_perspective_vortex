import numpy as np
from sklearn.covariance import EmpiricalCovariance
import scipy.linalg as la

class CSP :
    def __init__(self):
        """
        Source code :
                https://www.researchgate.net/publication/6524218_Common_Spatial_Pattern_Method_for_Channel_Selelction_in_Motor_Imagery_Based_Brain-computer_Interface
                https://www.researchgate.net/publication/304492710_Common_spatial_pattern-based_feature_extraction_from_the_best_time_segment_of_BCI_data
        """
        self.n_components = 4
        self.log = True
    
    def processReshape(self, matrice):
        nb_chan = matrice.shape[1]

        #### On change l'ordre des dimensions pour mettre les channels en premier
        matrice = np.transpose(matrice, [1, 0, 2])

        #### On tranforme X de 3 a 2 dimmensions
        matrice = matrice.reshape(nb_chan, -1)
        return matrice

    def normalizedCov(self, X, event):
        X = self.processReshape(X)
        # Normalized covariance of X (2D ndarray)
        cov = np.dot(X,np.transpose(X)) / np.trace(np.dot(X,np.transpose(X)))
        return cov

    def getAndStackCovs(self, X, y):
        """
        get the covariance matrix for each event of shape (n_channel, n_channel) and stack them.

        """
        covs = []
        for event in self.events:
            cov = self.normalizedCov(X[y == event], event)
            covs.append(cov)
        return np.stack(covs)

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        """
        #### Get a list of different events for the whole Data (here we have 2 events T1 T2)
        self.events = np.unique(y)

        self.X_total = X
        self.y_total = y

        #### transforme X in an array of stacked covariance matrix of each group of event T1 T2 (2 stacked covariance matrix)
        covs = self.getAndStackCovs(X, y)
        #### get eigen vector and eigen value from all cov matrix
        eigen_values, eigen_vectors = la.eigh(covs[0], covs.sum(0))
        order = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        #### we order the eigen_vectors in the matrice
        eigen_vectors = eigen_vectors[:, order]
        #### finaly get the filter
        self.filters_ = eigen_vectors.T
        self.patterns_ = la.pinv2(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the CSP filters.
        Parameters
        ----------
        X : data, array, shape (n_epochs, n_channels, n_times)
        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_sources)
            If self.transform_into == 'csp_space' then returns the data in CSP
            space and shape is (n_epochs, n_sources, n_times).
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # mean band power
        X = (X ** 2).mean(axis=2)
        if self.log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
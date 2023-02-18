import cv2
import numpy as np
from NearestNeighbor import KNearestNeighbor

class PCA(object):
    def __init__(self):
        self.m = None # mean
        self.K = 0  # K largest eigenvectors
        self.Wk = None # PCA Feature set (eigenvectors of covariance matrix)
        self.NNmodel = None # Nearest Neighbor

    def train(self, train_x, train_l, K):
        print("Performing PCA")
        self.K = K              # K largest eigenvectors
        X = train_x
        self.m = np.mean(X, axis=1)     # Global mean
        X = X - self.m.reshape(-1,1)    # Zero mean img vectors
        # Using the trick to compute eigenvectors faster
        u,s,v = np.linalg.svd(np.dot(X.T,X))
        # w = Xu 
        w = np.dot(X, v.T)
        w = w / np.linalg.norm(w, axis = 0)

        # Taking the first K eigenvectors
        self.Wk = w[:,:self.K]
        # Project training data into the eigenspace
        Y = np.dot(self.Wk.T, (X))
        # k-Nearest Neigbors model of the data
        self.NNmodel = KNearestNeighbor()
        self.NNmodel.fit(Y.T, train_l)

    def test(self, test_x, test_l):
        # Assign training vectors to X
        X = test_x
        # Project testing data into eigenspace
        Y = np.dot(self.Wk.T, X - self.m.reshape(-1,1))
        # Predict labels with k-Nearest Neighbors 
        predicted_labels = self.NNmodel.kneighbors(Y.T)
        # Accuracy
        acc = np.sum((predicted_labels - test_l) == 0) / np.float(test_l.size)
        return acc



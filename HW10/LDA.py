import cv2
import numpy as np
from NearestNeighbor import KNearestNeighbor

class LDA(object):
    def __init__(self):
        self.m = None # mean
        self.K = 0  # K largest eigenvectors
        self.Wk = None # PCA Feature set (eigenvectors of covariance matrix)
        self.NNmodel = None # Nearest Neighbor

    def train1(self, train_x, train_l, K):
        print("Performing LDA")
        # Get labels
        class_labels, counts = np.unique(train_l, return_counts=True)
        C = len(class_labels) # num of labels
        self.K = C-1 if K > C-1 else K  # At most eigenvectors C-1
        X = train_x
        # Construct S_W (within class scatter) matrix
        self.m = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))    # Global mean
        mc = np.zeros((X.shape[0], C))                              # class mean
        mi_m = np.zeros((X.shape[0],X.shape[1]))                    # Between class mean
        for i,c in enumerate(class_labels):
            cols = (train_l==c).flatten()
            X_c = X[:, cols]
            m_c = np.mean(X_c, axis=1)
            mc[:,i] = m_c
            mi_m[:,cols] = X_c - np.reshape(m_c, (X.shape[0], 1))

        # Construct S_B matrix  (between class scatter)
        xMat = mc - self.m
        # Diagolize S_B with eigendecomposition
        _,S,V = np.linalg.svd(np.dot(xMat.T,xMat))
        eigMat = np.eye(C)*S
        DB = np.sqrt(np.linalg.inv(eigMat))
        # Construct matrix Z = Y(DB)^-1/2
        Y = np.dot(xMat, V.T)
        Z = np.dot(Y, DB)
        # eigendecomposition of Z^T * SW * Z
        x_z = np.dot(np.dot(Z.T, mi_m), np.transpose(np.dot(Z.T, mi_m)))
        _,S,U = np.linalg.svd(x_z)
        # LDA eigenvectors that maximize the Fisher discriminant function
        w_z = np.dot(Z, U.T)
        w_z = w_z / np.linalg.norm(w_z, axis = 0)
        # Take the K largest
        self.Wk = w_z[:,:self.K]

        # Project training data into the eigenspace
        Y = np.dot(self.Wk.T, (X - self.m.reshape(-1,1)))
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
            
        
            




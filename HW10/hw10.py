import cv2 as cv
import numpy as np
from PCA import PCA
from LDA import LDA
import os
import matplotlib.pyplot as plt
from NearestNeighbor import KNearestNeighbor
from adaboost import CascadedAdaboost

###============================Functions===========================###
def loadData(data_path, dim, uglabels = None):
    imgList = [img for img in os.listdir(data_path) if img.endswith(".png")]
    num_imgs = len(imgList)
    X = np.zeros((dim[0]*dim[1], num_imgs), dtype=float)
    labels = np.zeros((num_imgs, 1), dtype=float)
    for i,imgName in enumerate(imgList):
        img = cv.imread(os.path.join(data_path, imgName), 0)
        X[:,i] = img.flatten()
        if uglabels == None:
            labels[i] = int(imgName.split('_')[0])
        if uglabels != None:
            labels[i] = uglabels
    # Normalize data
    X_mean = np.mean(X)
    X_std = np.std(X)
    X = X - X_mean
    X = X / X_std 
    return X, labels

def Dim_Red(train_x, train_l, test_x, test_l, algos, maxK):
    accuracies = []
    for K in range(1,maxK+1):
        print(f'K: {K}')
        if 'pca' in algos:
            pca = PCA()
            pca.train(train_x, train_l, K)
            pca_acc = pca.test(test_x, test_l)
            print(f'PCA Accuracy: {pca_acc}')
            accuracies.append(pca_acc)
        if 'lda' in algos:
            lda = LDA()
            lda.train1(train_x, train_l, K)
            lda_acc = lda.test(test_x, test_l)
            print(f'LDA Accuracy: {lda_acc}')
            accuracies.append(lda_acc)
    if len(algos) == 2:
        accuracies = np.reshape(accuracies, (maxK, 2))
    elif len(algos) == 1:
        accuracies = np.reshape(accuracies, (maxK, 1))
    return np.asarray(accuracies)

def plotAccuracies(accuracies):
    maxK = accuracies.shape[0]
    xdata = np.linspace(0,maxK, num=maxK)
    acc_plt = plt.figure()
    line1, = plt.plot(xdata, accuracies[:,0], '-ro', label = 'PCA', linewidth=1)
    line2, = plt.plot(xdata, accuracies[:,1], '-go', label = 'LDA', linewidth=1)
    return acc_plt, line1, line2

def plotTask3res(train_res, test_res, name, loc):
    plt.figure()
    line1, = plt.plot(np.arange(1, len(train_res)+1), train_res, '-go', label = 'Train')
    line2, = plt.plot(np.arange(1, len(test_res)+1), test_res, '-bo', label = 'Test')
    plt.legend(handles=[line1, line2], loc=loc)
    plt.xlabel('Cascaded Adaboost Classifier (N)')
    plt.ylabel(name)
    plt.title(name)
    plt.show()

###=========================Main Function==========================###
## Load data
print("Task 1 and Task 2")
print("Loading data...")
train_x, train_l = loadData('HW10/FaceRecognition/train/', (128,128))
print(f'Training Data: {train_x.shape}, Training Labels: {train_l.shape}')

test_x, test_l = loadData('HW10/FaceRecognition/test/', (128,128))
print(f'Test Data: {test_x.shape}, Test Labels: {test_l.shape}')

# ### Task 1: Face Recognition with PCA, LDA
print("Starting PCA and LDA")
algos = ['pca', 'lda']
accuracies = Dim_Red(train_x, train_l, test_x, test_l, algos, maxK = 20)
plot, line1, line2 = plotAccuracies(accuracies)

### Task 2: Face Recognition with Autoencoders
print("Starting Autoencoders")
from autoencoder import autoencoderOutput

P = [3, 8, 16]
acc_autoencoder = np.zeros(len(P))
for i,p in enumerate(P):
    print(f'P:{p}')
    X_train, y_train, X_test, y_test = autoencoderOutput(p)
    
    # Fit k-Nearest neighbors and predict
    NNmodel = KNearestNeighbor(k_neighbors = 1)
    NNmodel.fit(X_train, np.reshape(y_train, (y_train.shape[0], 1)))
    predicted_labels = NNmodel.kneighbors(X_test)
    acc_autoencoder[i] = np.sum((predicted_labels - np.reshape(y_train, (y_train.shape[0], 1))) == 0) / y_test.size

# Plot the accuracies on the same plot
line_ae_3, = plt.plot(3, acc_autoencoder[0], '-bo', label = 'AutoEncoder_3')
line_ae_8, = plt.plot(8, acc_autoencoder[1], '-co', label = 'AutoEncoder_8')
line_ae_16, = plt.plot(16, acc_autoencoder[2], '-yo', label = 'AutoEncoder_16')

plt.legend(handles=[line1, line2, line_ae_3, line_ae_8, line_ae_16], loc=4)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

### Task 3: Adaboost Classifier
# Load data
print("Starting Adaboost")
print("Loading data")
train_x_pos, train_l_pos = loadData('HW10/CarDetection/train/positive', (40,20), 1)
train_x_neg, train_l_neg = loadData('HW10/CarDetection/train/negative', (40,20), 0)
train_x = np.hstack((train_x_pos, train_x_neg))
train_l = np.hstack((train_l_pos.T, train_l_neg.T))
print(f'Training Data: {train_x.shape}, Training Labels: {train_l.shape}')

test_x_pos, test_l_pos = loadData('HW10/CarDetection/test/positive', (40,20), 1)
test_x_neg, test_l_neg = loadData('HW10/CarDetection/test/negative', (40,20), 0)
test_x = np.hstack((test_x_pos, test_x_neg))
test_l = np.hstack((test_l_pos.T, test_l_neg.T))
print(f'Test Data: {test_x.shape}, Test Labels: {test_l.shape}')

num_cascades = 10
num_classifier_per_cascade = 25
carClassifier = CascadedAdaboost(train_x, train_l, test_x, test_l)
FP_train, FN_train, TP_train, Ac_train, FP_test, FN_test, TP_test, Ac_test = carClassifier.train(num_cascades, num_classifier_per_cascade)

# Plot results
plotTask3res(FP_train, FP_test, 'False Positive Rate', 1)
plotTask3res(FN_train, FN_test, 'False Negative Rate', 7)
plotTask3res(TP_train, TP_test, 'True Positive Rate', 7)
plotTask3res(Ac_train, Ac_test, 'Accuracy', 4)


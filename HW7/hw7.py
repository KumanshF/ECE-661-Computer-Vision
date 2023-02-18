import cv2
import math
import numpy as np
import BitVector as bt
from vgg import VGG19
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sn


#---------------------------Functions-----------------------------#
def getFeatureMatrix_LBP(image, R, P):
    '''
    This function is inspired by Prof. Kak's 
    python code to get LBP descriptor vector
    Source code found at: 
    https://engineering.purdue.edu/kak/Tutorials/TextureAndColor.pdf

    BitVector Module by Prof. Kak is also used to compute 
    the encodings for the patterns
    '''
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64),interpolation = cv2.INTER_AREA)

    binaryHist = np.zeros((P+2))
    totalPixels = img.shape[0]*img.shape[1]

    for i in range(R, img.shape[1]-R):
        for j in range(R, img.shape[0]-R):
            pattern = []
            for p in range(P):
                del_k,del_l = R*math.cos(2*math.pi*p/P), R*math.sin(2*math.pi*p/P)
                if abs(del_k) < 0.001: del_k = 0.0
                if abs(del_l) < 0.001: del_l = 0.0
                k, l = i + del_k, j + del_l
                k_b,l_b = int(k),int(l)
                delta_k,delta_l = k-k_b,l-l_b
                if (delta_k < 0.001) and (delta_l < 0.001):
                    pVal = float(img[k_b][l_b])
                elif (delta_l < 0.001):
                    pVal = (1 - delta_k) * img[k_b][l_b] + \
                        delta_k * img[k_b+1][l_b]
                elif (delta_k < 0.001):
                    pVal = (1 - delta_l) * img[k_b][l_b] + \
                        delta_l * img[k_b][l_b+1]
                else:
                    pVal = (1-delta_k)*(1-delta_l)*img[k_b][l_b] \
                        + (1-delta_k)*delta_l*img[k_b][l_b+1] \
                        + delta_k*delta_l*img[k_b+1][l_b+1] \
                        + delta_k*(1-delta_l)*img[k_b+1][l_b]
                if pVal >= img[i][j]:
                    pattern.append(1)
                else:
                    pattern.append(0)
            bv = bt.BitVector(bitlist = pattern)
            intVals = [int(bv<<1) for _ in range(P)]
            minbv = bt.BitVector(intVal = min(intVals), size = P)
            runs = minbv.runs()
            if len(runs)>2:
                binaryHist[P+1] += 1
            elif len(runs)==1 and runs[0][0]=='1':
                binaryHist[P] += 1
            elif len(runs)==1 and runs[0][0]=='0':
                binaryHist[0] += 1
            else:
                binaryHist[len(runs[1])] += 1
    featureVector = binaryHist/totalPixels
    return featureVector

def getFeatureMatrix_Gram(img, vgg, cl, i):
    featureMat = []
    featureMap = vgg(img)
    for r in range(featureMap.shape[0]):
        mat = featureMap[r]
        featureMat.append(mat.flatten())
    bigGramMat = np.matmul(featureMat, np.transpose(featureMat)) # 512x512 matrix
    uppertriangleIdx = np.triu_indices(bigGramMat.shape[1])      # Get upper-triangle indices
    upperTriangleGramMat = bigGramMat[uppertriangleIdx]          # Get upper-triangle of Gram Matrix   
    upperTriangleGramMat = np.ravel(upperTriangleGramMat)        # Flatten the matrix
    indx = int(len(upperTriangleGramMat)/2)
    featureVector = upperTriangleGramMat[indx:(indx+1024)]      # Select random 1024 elements

    # Plot the Gram Matrix for each class
    if i==1:
        # scale the values for better visual
        bigGramMat = bigGramMat+0.001
        cmap = plt.cm.gray
        norm = colors.LogNorm()
        plot = cmap(norm(bigGramMat))
        # Save plot with the name of the image
        filename = 'HW7/plots/'+str(cl)+str(i)+'.jpg'
        plt.imsave(fname=filename, arr=plot, format = 'jpg')
    return featureVector

def extractFeature_LBP(dataSetnums, path, labels):
    featureMatrix = []
    labelMatrix = []
    # Iterate through all training images
    for key,num in dataSetnums.items():
        cl = labels[key]
        print("Working on class:", key, cl)
        for i in range(num[0],num[1]+1):
            imgName = str(cl)+str(i)
            img = io.imread(str(path)+imgName+".jpg")
            if img is None or len(img.shape)!=3 or (len(img.shape)==3 and img.shape[2]!=3):
                continue
            img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)

            # LBP feature Extraction
            featureVector = getFeatureMatrix_LBP(img, R = 1, P = 8)
            print(imgName)
            featureMatrix.append(featureVector)
            labelMatrix.append(key)

            # Plot the histogram
            if i == 1:
                print(featureVector)
                plt.figure()
                cm = plt.cm.get_cmap('tab20c')
                x = np.random.rand()
                plt.bar(range(len(featureVector)), featureVector, width = 0.8,color = cm(x), edgecolor = 'black')
                plt.title('LBP Histogram of '+str(imgName)+'.jpg')
                plt.savefig('HW7/plots/'+str(imgName)+'_hist.jpg')
                
    return featureMatrix, labelMatrix 

def extractFeatures_Gram(dataSetnums, path, labels):
    gramMatrix = []
    labelMatrix = []
    # Load the model and the provided pretrained weights
    vgg = VGG19()
    vgg.load_weights('HW7/vgg_normalized.pth')
    # Iterate through all training images
    for key,num in dataSetnums.items():
        cl = labels[key]
        print("Working on class:", key, cl)
        for i in range(num[0],num[1]+1):
            imgName = str(cl)+str(i)
            img = io.imread(str(path)+imgName+".jpg")
            if img is None or len(img.shape)!=3 or (len(img.shape)==3 and img.shape[2]!=3):
                continue
            img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)

            # Gram Matrix feature Extraction
            gramFeatureVector = getFeatureMatrix_Gram(img, vgg, cl, i)
            print(imgName) 
            gramMatrix.append(gramFeatureVector)
            labelMatrix.append(key)
    return gramMatrix, labelMatrix

def trainSVM(trainingData, labels):
    # Train SVM multi-class classifier
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_INTER)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 110, 1e-6))
    svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)
    return svm

def predictLabels(trainedSVM, data):
    labels = trainedSVM.predict(data)[1]
    return labels

def getConfusionMatrix(predictedLabels, actualLabels):
    confusionMatrix = np.zeros((4,4))
    for i in range(len(predictedLabels)):
        predictedLabel = int(predictedLabels[i,0]) - 1
        actualLabel = actualLabels[i] - 1
        confusionMatrix[predictedLabel, actualLabel] += 1
    return confusionMatrix

def drawConfusionMatrix(cmat, method):
    plot_cm = pd.DataFrame(cmat, index = ["Cloudy", "Rain", "Shine", "Sunrise"], columns = ["Cloudy", "Rain", "Shine", "Sunrise"]) 
    plt.figure()
    sn.heatmap(plot_cm, vmax = 50, annot=True, cmap=sn.color_palette("Blues", as_cmap=True))
    plt.xlabel("Actual Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Confusion Matrix for "+str(method)+" Method")
    plt.savefig('HW7/plots/'+str(method)+'.jpg')
    return 0

#---------------------------Main Code-----------------------------#
labels = {1: "cloudy", 2: "rain", 3:"shine", 4:"sunrise"}
numTrainingImages = {1: [1,250], 2: [1,165], 3: [1,203], 4: [1,307]}
numTestingImages = {1: [251, 300], 2: [166,215], 3: [204,253], 4: [308,357]}

#============Actions=============#
ExtractFeature = False
TrainAndTest = True
#================================#

if ExtractFeature: 
    # Extracting Texture Descriptors for training data 
    featureMatrix_Training_lbp, labelMatrix_Training_lbp = extractFeature_LBP(numTrainingImages, "HW7/data/training/", labels)
    np.savez_compressed('HW7/data/featureMatrix_Training_lbp', featureMat = featureMatrix_Training_lbp, labels = labelMatrix_Training_lbp)

    gramMatrix_Training_cnn, labelMatrix_Training_cnn = extractFeatures_Gram(numTrainingImages, "HW7/data/training/", labels)
    np.savez_compressed('HW7/data/featureMatrix_Training_cnn', featureMat = gramMatrix_Training_cnn, labels = labelMatrix_Training_cnn)
    
    # Extracting Texture Descriptors for testing data
    featureMatrix_Testing_lbp, labelMatrix_Testing_lbp = extractFeature_LBP(numTestingImages, "HW7/data/testing/", labels)
    np.savez_compressed('HW7/data/featureMatrix_Testing_lbp', featureMat = featureMatrix_Testing_lbp, labels = labelMatrix_Testing_lbp)

    gramMatrix_Testing_cnn, labelMatrix_Testing_cnn = extractFeatures_Gram(numTestingImages, "HW7/data/testing/", labels)
    np.savez_compressed('HW7/data/featureMatrix_Testing_cnn', featureMat = gramMatrix_Testing_cnn, labels = labelMatrix_Testing_cnn)


if TrainAndTest:
    #============================== TRAIN ===============================# 
    # load the feature matrices for lbp method
    trainingData_lbp = np.load('HW7/data/featureMatrix_Training_lbp.npz')
    featureMatrix_Training_lbp = np.matrix(trainingData_lbp['featureMat'], dtype=np.float32)
    labelMatrix_Training_lbp = np.array(trainingData_lbp['labels'])

    # Load the feature matrices for gram matrix method
    trainingData_cnn = np.load('HW7/data/featureMatrix_Training_cnn.npz')
    gramMatrix_Training_cnn = np.matrix(trainingData_cnn['featureMat'], dtype=np.float32)
    labelMatrix_Training_cnn = np.array(trainingData_cnn['labels'])

    # Get trained SVM using training data and labels
    svm_lbp = trainSVM(featureMatrix_Training_lbp, labelMatrix_Training_lbp)
    svm_cnn = trainSVM(gramMatrix_Training_cnn, labelMatrix_Training_cnn)


    #=============================== TEST ===============================#
    # Get the test data for lbp method
    testingData_lbp = np.load('HW7/data/featureMatrix_Testing_lbp.npz')
    featureMatrix_Testing_lbp = np.matrix(testingData_lbp['featureMat'], dtype=np.float32)
    labelMatrix_Testing_lbp = np.array(testingData_lbp['labels'])

    # Get the test data for gram matrix method
    testingData_cnn = np.load('HW7/data/featureMatrix_Testing_cnn.npz')
    gramMatrix_Testing_cnn = np.matrix(testingData_cnn['featureMat'], dtype=np.float32)
    labelMatrix_Testing_cnn = np.array(testingData_cnn['labels'])
    
    # Predict labels for test data
    predictedLabels_lbp = predictLabels(svm_lbp, featureMatrix_Testing_lbp)
    predictedLabels_cnn = predictLabels(svm_cnn, gramMatrix_Testing_cnn)

    # Build the confusion matrix
    confusionMatrix_lbp = getConfusionMatrix(predictedLabels_lbp, labelMatrix_Testing_lbp)
    confusionMatrix_cnn = getConfusionMatrix(predictedLabels_cnn, labelMatrix_Testing_cnn)

    print(confusionMatrix_lbp)
    print(confusionMatrix_cnn)

    drawConfusionMatrix(confusionMatrix_lbp, "LBP")
    drawConfusionMatrix(confusionMatrix_cnn, "GramMatrix")

    
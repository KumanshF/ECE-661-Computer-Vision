import cv2 as cv
import numpy as np

IMG_H = 20
IMG_W = 40
NUM_FEATURES = 47232

def get_ftMat():
    '''
        Calculate the feature matrix to compute the feature vectors
    '''
    # ftMat was initiallized with a very larger number (50000, 20*80) first 
    # and NUM_features was then updated
    ftMat = np.zeros((NUM_FEATURES, IMG_H*IMG_W), dtype=np.int8)
    ftNum = 0
    offset = 2
    # Horizontal filter
    filter_w, filter_h = 2, 1
    for i in range(1, IMG_H, filter_h):
        for j in range(1, IMG_W, filter_w):
            for y in range(offset, IMG_H - filter_h*i + 1 - offset):
                for x in range(offset, IMG_W - filter_w*j + 1 - offset):
                    ftMat[ftNum, y*IMG_H + x] = 1.0
                    ftMat[ftNum, y*IMG_H + filter_w*j//2] = -2.0
                    ftMat[ftNum, y*IMG_H + filter_w*j] = 1.0
                    ftMat[ftNum, (y+filter_w*i)*IMG_H+x] = -1.0
                    ftMat[ftNum, (y+filter_w*i)*IMG_H+x+filter_w*j//2] = 2.0
                    ftMat[ftNum, (y+filter_w*i)*IMG_H+x+filter_w*j] = -1.0
                    ftNum += 1
    # Vertical filter
    filter_w, filter_h = 1, 2
    for i in range(1, IMG_H, filter_h):
        for j in range(1, IMG_W, filter_w):
            for y in range(offset, IMG_H - filter_h*i + 1 - offset):
                for x in range(offset, IMG_W - filter_w*j + 1 - offset):
                    ftMat[ftNum, y*IMG_H + x] = -1.0
                    ftMat[ftNum, y*IMG_H + filter_w*j] = 1.0
                    ftMat[ftNum, (y+filter_h*i//2)*IMG_H+x] = 2.0
                    ftMat[ftNum, (y+filter_h*i//2)*IMG_H+x+filter_w*j] = -2.0
                    ftMat[ftNum, (y+filter_h*i)*IMG_H+x//2] = -1.0
                    ftMat[ftNum, (y+filter_h*i)*IMG_H+x+filter_w*j] = 1.0
                    ftNum += 1
    # print(ftNum)
    return ftMat

class CascadedAdaboost(object):
    '''
        Inspired from 
        1. https://github.com/lifangda01/ECE661-ComputerVision/blob/master/HW11/
        2. https://medium.com/@rohan.chaudhury.rc/adaboost-classifier-for-face-
                detection-using-viola-jones-algorithm-30246527db11
    '''
    def __init__(self, train_X, train_l, test_X, test_l):
        self.train_X = train_X
        self.train_l = train_l
        self.test_X = test_X
        self.test_l = test_l
        self.train_posNum =  0
        self.train_negNum =  0
        self.ftMat = get_ftMat()
        self.cascaded_classifiers = []
        self.train_fvecs = np.dot(self.ftMat, self.train_X)    # Feature Vector (47232 x 2468)
        
        
    def train(self, num_cascades, num_classifier):
        # Sort data
        self.train_posNum = int(np.sum(self.train_l))                # Num of neg training data
        self.train_negNum = self.train_l.size - self.train_posNum    # Num of pos training data
        self.train_num = self.train_l.size                           # Num of total training data

        # Seperate pos/neg data feature vectors
        all_pos_fvecs = self.train_fvecs[:,(self.train_l==1).flatten()]
        all_neg_fvecs = self.train_fvecs[:,(self.train_l==0).flatten()]
        current_pos_fvecs = all_pos_fvecs
        current_neg_fvecs = all_neg_fvecs
        
        FP_train = []       # False Pos Rate
        FN_train = []       # False Neg Rate
        TP_train = []       # True Pos rate
        Ac_train = []       # Accuracy
        FP_test = []
        FN_test = []
        TP_test = []
        Ac_test = []
        
        # Cascade adaboost classifiers
        for i in range(num_cascades):
            print(f'Training {i+1} Adaboost classifier in the Cascade')
            current_adaboost = self.addAdaboostClassifier()
            current_adaboost.set_training_fvecs(current_pos_fvecs, current_neg_fvecs)
            for j in range(num_classifier):
                print(f"Adding weak classifier {j+1}")
                current_adaboost.addWeakClassifier()
            falsePos_ind, FP, FN, TP, Ac = self.classifyTrainingData()
            FP_train.append(FP)
            FN_train.append(FN)
            TP_train.append(TP)
            Ac_train.append(Ac)
            current_neg_fvecs = all_neg_fvecs[:, falsePos_ind - self.train_posNum]
            FP, FN, TP, Ac = self.classifyTestingData()
            FP_test.append(FP)
            FN_test.append(FN)
            TP_test.append(TP)
            Ac_test.append(Ac)
        return FP_train, FN_train, TP_train, Ac_train, FP_test, FN_test, TP_test, Ac_test

    def addAdaboostClassifier(self):
        classifier = AdaboostClassifier()
        classifier.setftMat(self.ftMat)
        self.cascaded_classifiers.append(classifier)
        return classifier
    
    def classifyTrainingData(self):
        print("Classifying training images")
        ftVecs = self.train_fvecs
        pos_inds = np.arange(self.train_num)
        for c in self.cascaded_classifiers:
            predictions = c.classify_ftVecs(ftVecs)
            ftVecs = ftVecs[:, predictions==1]
            pos_inds = pos_inds[predictions==1]
        # Sort TP, FP, FN
        FP_inds = pos_inds[np.take(self.train_l, pos_inds)==0]
        TP_num = np.sum(np.take(self.train_l, pos_inds))
        TP = TP_num / self.train_posNum
        FP = (pos_inds.size - TP_num) / self.train_negNum
        FN =  1 - TP
        w = self.train_posNum / (self.train_posNum + self.train_negNum) # weight
        Ac = TP * w + (1-FP)*(1-w)
        print("FP = %.4f, FN = %.4f, TP = %.4f, Acc = %.4f" % (FP, FN, TP, Ac))
        return FP_inds, FP, FN, TP, Ac
    
    def classifyTestingData(self):
        print("Classifying test images")
        ftVecs = np.dot(self.ftMat, self.test_X)
        test_posNum = int(np.sum(self.test_l))          # Num of neg test data
        test_negNum = self.test_l.size - test_posNum    # Num of pos test data
        test_num = self.test_l.size                     # Num of total test data
        pos_inds = np.arange(test_num)
        for c in self.cascaded_classifiers:
            predictions = c.classify_ftVecs(ftVecs)
            ftVecs = ftVecs[:, predictions==1]
            pos_inds = pos_inds[predictions==1]
        TP_num = np.sum(np.take(self.test_l, pos_inds))
        TP = TP_num / test_posNum
        FP = (pos_inds.size - TP_num) / test_negNum
        FN =  1 - TP
        w = test_posNum / (test_posNum + test_negNum)
        Ac = TP * w + (1-FP)*(1-w)
        print("FP = %.4f, FN = %.4f, TP = %.4f, Acc = %.4f" % (FP, FN, TP, Ac))
        return FP, FN, TP, Ac

class AdaboostClassifier():
    def __init__(self):
        self.train_l = None
        self.train_sortedInds = None
        self.train_ftVecs = None
        self.train_posNum = 0
        self.train_negNum = 0
        self.threshold = 1.0
        self.sample_weights = None
        self.weakClassifier_inds = np.array([], dtype=int)
        self.weakClassifier_polarities = np.array([])
        self.weakClassifier_threshs = np.array([])
        self.weakClassifier_weights = np.array([])
        self.weakClassifier_results = np.array([])
        self.weakClassifier_weighted_results = None

    def setftMat(self, ftMat):
        self.ftMat = ftMat
    
    def set_training_fvecs(self, pos_fvecs, neg_fvecs):
        print("Sorting current training features")
        self.train_posNum = pos_fvecs.shape[1]
        self.train_negNum = neg_fvecs.shape[1]
        self.train_l = np.hstack((np.ones(self.train_posNum), np.zeros(self.train_negNum)))
        self.train_ftVecs = np.hstack((pos_fvecs, neg_fvecs))
        self.train_sortedInds = np.argsort(self.train_ftVecs, axis=1)
        print(f'Pos sample: {self.train_posNum}, Neg sample: {self.train_negNum}')

    def addWeakClassifier(self):
        if self.weakClassifier_inds.size == 0:
            self.sample_weights = np.zeros(self.train_l.size, dtype=float)
            self.sample_weights.fill(1.0 / (2*self.train_negNum))
            self.sample_weights[self.train_l==1] = 1.0 / (2*self.train_posNum)
        else:
            self.sample_weights = self.sample_weights / np.sum(self.sample_weights)
        # Get the best classifier with the minimum error
        best_ft_ind, best_ft_pol, best_ft_thresh, best_ft_error, best_ft_results = self.getBestWeakClassifier()
        self.weakClassifier_inds = np.append(self.weakClassifier_inds, best_ft_ind)
        self.weakClassifier_polarities = np.append(self.weakClassifier_polarities, best_ft_pol)
        self.weakClassifier_threshs = np.append(self.weakClassifier_threshs, best_ft_thresh)
        # calculate confidence
        beta = best_ft_error / (1 - best_ft_error)
        # Trust Factor
        alpha = np.log(1 / np.abs(beta))
        self.weakClassifier_weights = np.append(self.weakClassifier_weights, alpha)
        e = np.abs(best_ft_results - self.train_l)
        # Update the weight
        self.sample_weights = self.sample_weights*beta**(1-e)
        # Adjust the threshold
        if self.weakClassifier_results.size == 0:
            self.weakClassifier_results = best_ft_results.reshape(-1,1)
        else: 
            self.weakClassifier_results = np.hstack((self.weakClassifier_results, best_ft_results.reshape(-1,1)))
        self.weakClassifier_weighted_results = np.dot(self.weakClassifier_results, self.weakClassifier_weights)
        self.threshold = min(self.weakClassifier_weighted_results[self.train_l==1])
    
    def getBestWeakClassifier(self):
        ft_errs = np.zeros(NUM_FEATURES)
        ft_thresh = np.zeros(NUM_FEATURES)
        ft_pol = np.zeros(NUM_FEATURES)
        ft_sorted_ind = np.zeros(NUM_FEATURES, dtype=int)
        Tplus = np.sum(self.sample_weights[self.train_l==1])
        Tminus = np.sum(self.sample_weights[self.train_l==0])
        for r in range(NUM_FEATURES):
            # Get weights of sorted feature vectors
            sorted_weights = self.sample_weights[self.train_sortedInds[r,:]]
            sorted_l = self.train_l[self.train_sortedInds[r,:]]
            Splus = np.cumsum(sorted_l * sorted_weights)
            Sminus = np.cumsum(sorted_weights) - Splus
            Eplus = Splus + Tminus - Sminus
            Eminus = Sminus + Tplus - Splus
            # Calculate polarity
            polarities = np.zeros(self.train_posNum + self.train_negNum)
            polarities[Eplus > Eminus] = -1
            polarities[Eplus <= Eminus] = 1
            # Get errors
            errs = np.minimum(Eplus, Eminus)
            sorted_ind = np.argmin(errs)
            min_err_sample_ind = self.train_sortedInds[r,sorted_ind]
            min_err = np.min(errs)
            # Get threshold based on min err index
            threshold = self.train_ftVecs[r,min_err_sample_ind]
            polarities = polarities[sorted_ind]
            ft_errs[r] = min_err
            ft_thresh[r] = threshold
            ft_pol[r] = polarities
            ft_sorted_ind[r] = sorted_ind
        best_ft_ind = np.argmin(ft_errs)
        best_ft_pol = ft_pol[best_ft_ind]
        best_ft_thresh = ft_thresh[best_ft_ind]
        best_ft_error = ft_errs[best_ft_ind]
        best_ft_results = np.zeros(self.train_posNum + self.train_negNum)
        best_sorted_ind = ft_sorted_ind[best_ft_ind]
        if best_ft_pol == 1:
            best_ft_results[self.train_sortedInds[best_ft_ind, best_sorted_ind:]] = 1
        else:
            best_ft_results[self.train_sortedInds[best_ft_ind, :best_sorted_ind]] = 1
        
        return best_ft_ind, best_ft_pol, best_ft_thresh, best_ft_error, best_ft_results
    

    def classify_ftVecs(self, ftVecs):
        weakClassifiers = ftVecs[self.weakClassifier_inds,:]
        pol_vec = self.weakClassifier_polarities.reshape(-1,1)
        thresh_vec = self.weakClassifier_threshs.reshape(-1,1)
        # Predictions of weak classifiers
        weakClassifiers_preds = weakClassifiers * pol_vec > thresh_vec * pol_vec
        weakClassifiers_preds[weakClassifiers_preds==True] = 1
        weakClassifiers_preds[weakClassifiers_preds==False] = 0
        # Apply weights
        finalClassifier_result = np.dot(self.weakClassifier_weights, weakClassifiers_preds)
        final_preds = np.zeros(finalClassifier_result.size)
        final_preds[finalClassifier_result >= self.threshold] = 1
        return final_preds

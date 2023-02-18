#####===========Task 3: Dense Stereo Matching==========#####

import numpy as np
import cv2 as cv

###============================Functions===========================###
def getDispMap(imageL, imageR, M, dMax):
    imgL = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY) if len(imageL.shape)==3 else imageL
    imgR = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY) if len(imageR.shape)==3 else imageR
    
    win = int(M/2)
    rg = dMax+win
    w, h = imgL.shape[1], imgL.shape[0]
    dMap = np.zeros((h,w), dtype=np.uint8)
    for rowL in range(rg, h-rg):
        print(f'Row: {rowL+1}/{h-rg}')
        for colL in range(w-rg-1, rg-1, -1): #go right to left
            dataCost = []
            winL = imgL[rowL-win:rowL+win+1, colL-win:colL+win+1]
            binL = np.ravel(np.where(winL>imgL[rowL, colL], 1, 0))
            for d in range(dMax+1):
                colR = colL-d
                winR = imgR[rowL-win:rowL+win+1, colR-win:colR+win+1]
                binR = np.ravel(np.where(winR>imgR[rowL, colR], 1, 0))
                cost = np.bitwise_xor(binL, binR)
                cost = np.sum(cost)
                dataCost.append(cost)
            dMap[rowL,colL] = np.argmin(dataCost) # d for which cost in min
    dMap = dMap.astype(np.uint8)
    print("Max dMap value:", np.max(dMap))
    # get Mask
    dMapMask = cv.normalize(dMap, dst=None, alpha=0, beta=255, norm_type = cv.NORM_MINMAX).astype(np.uint8)
    return dMap, dMapMask, rg

def getAccuracy(gtDispMap, dMap, rg, sigma):
    # Create subarea to remove black borders from disparity maps
    subarea_gt = gtDispMap[rg:gtDispMap.shape[0]-rg, rg:gtDispMap.shape[1]-rg]
    subarea_dMap =  dMap[rg:dMap.shape[0]-rg, rg:dMap.shape[1]-rg]
    # Find the error difference
    diff = np.abs(subarea_dMap.astype(np.uint16) - subarea_gt.astype(np.uint16)).astype(np.uint8)
    N = cv.countNonZero(subarea_gt[:,:])
    Ntotal = subarea_gt.shape[1]*subarea_gt.shape[0]
    print(f'Valid Pixels N: {N}/{Ntotal}')
    # accuracy
    acc = np.sum(diff<=sigma)/N
    # Error Mask 
    errMask = np.where(diff<=sigma, 255, 0)
    
    return acc, errMask
###=========================Main Function==========================###
#Input images
imgL = cv.imread('HW9/task3/im2.ppm', 0)
imgR = cv.imread('HW9/task3/im6.ppm', 0)

# Input the left disparity map
gtDispMap = cv.imread('HW9/task3/disp2.pgm',0)
gtDispMap = gtDispMap.astype(np.float32) / 4.0
gtDispMap = gtDispMap.astype(np.uint8)
dMax = np.max(gtDispMap)
print(dMax)

# Use Census transform on left image to make its disparity map for different M
for M in [13, 18, 29, 35]:
    print(f'M: {M}')
    dMapL, dMapMaskL, rg = getDispMap(imgL, imgR, M, dMax)
    cv.imwrite('HW9/task3/dMapL'+str(M)+'.jpg', dMapMaskL)
    percL, errMaskL = getAccuracy(gtDispMap, dMapL, rg, 2)
    print(f'Accuracy percentage for M {M}: {percL}')
    cv.imwrite('HW9/task3/errMaskL'+str(M)+'.jpg', errMaskL)
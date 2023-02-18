import math
from pickletools import optimize
import numpy as np
import cv2 as cv
import random
from scipy import optimize
#---------------------------Functions-----------------------------#
def drawCorrespondences(corrs, img0, img1, numCorres):
    w = img0.shape[1]
    comb_img = np.concatenate((img0, img1), axis=1)
    for corr in corrs[0:numCorres]:
        x = corr[0]
        xp = [corr[1][0]+w, corr[1][1]]
        color = np.random.randint(0, 255, size=(3, ))
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv.circle(comb_img, tuple(x), 3, tuple(color), 1)
        cv.circle(comb_img, tuple(xp), 3, tuple(color), 1)
        cv.line(comb_img, tuple(x), tuple(xp), tuple(color), 1)
    return comb_img

def getDistance(des1, des2, mode):
    if mode == 'NCC':
        m1 = np.mean(des1)
        m2 = np.mean(des2)
        d = 1 - np.sum((des1-m1)*(des2-m2))/(np.sqrt(np.sum(np.square(des1-m1))*np.sum(np.square(des2-m2))))
    elif mode =='SSD':
        d = np.sum(np.square(des1-des2))
    return d

def sift(img1_raw, img2_raw):
    if len(img1_raw.shape) == 3:
        img1 = cv.cvtColor(img1_raw,cv.COLOR_BGR2GRAY)
    else:
        img1 = img1_raw

    if len(img2_raw.shape) == 3:
        img2 = cv.cvtColor(img2_raw,cv.COLOR_BGR2GRAY)
    else:
        img2 = img2_raw

    # Create sift detector
    sift_detector = cv.SIFT_create(5000)

    # Find key points and descriptors with Sift
    kp1, des1 = sift_detector.detectAndCompute(img1,None)
    kp2, des2 = sift_detector.detectAndCompute(img2,None)

    # Find Correspondences between key points 
    
    #---------Finding correspondence with SSD or NCC-------# 
    # corrs = []
    # for i in range(len(kp1)):
    #     distances = []
    #     domain_coord = list(map(int,kp1[i].pt))
    #     for j in range(len(kp2)):
    #         dist = getDistance(des1[i], des2[j], 'SSD')
    #         distances.append(dist)
    #     bestM = list(map(int, kp2[np.argmin(distances)].pt))
    #     corrs.append(np.around([domain_coord, bestM]).astype(int))

    #--------Finding Correspondences with BFMatcher-------#
    corrs = []
    bf = cv.BFMatcher()
    matches=bf.match(des1,des2)
    matches=sorted(matches, key= lambda x:x.distance)
    for mat in matches:
        domain_coord = kp1[mat.queryIdx].pt
        bestM = kp2[mat.trainIdx].pt
        corrs.append(np.around([domain_coord, bestM]).astype(int))
    return corrs

def estimateH(corrs):
    numCorrs = len(corrs)
    if numCorrs<4:
        return 0
    
     # A and B Matrix
    A = np.zeros((2*numCorrs,8))
    b = np.zeros((2*numCorrs,1))
    for i in range(numCorrs):
        pair = corrs[i]
        xp = pair[1][0]
        yp = pair[1][1]
        x = pair[0][0]
        y = pair[0][1]
        # A matrix
        A[2*i][0:8] = [x, y, 1 , 0 , 0 , 0 ,-x*xp, -y*xp]
        A[2*i+1][0:8] = [0, 0, 0, x, y, 1, -x*yp, -y*yp]  
        # B matrix
        b[2*i] = xp
        b[2*i+1] = yp
    
    # H matrix
    H = np.zeros((3,3))
    H = np.matmul(np.linalg.pinv(A),b)
    H = np.append(H,[1])
    H = np.reshape(H, (3,3))

    return H

def getInliers(H, corrs, delta):
    dom_cords = np.zeros((len(corrs), 3), np.uint16)
    range_cords = np.zeros((len(corrs), 3), np.uint16)
    for i in range(len(corrs)):
        pair = corrs[i]
        dom = pair[0]
        rang = pair[1]
        dom_cords[i] = [dom[0], dom[1], 1]
        range_cords[i] = [rang[0], rang[1], 1]
    Y = np.matmul(H, np.transpose(dom_cords))
    Y = Y / Y[2,:]

    err = np.square(range_cords.T - Y)
    dist = np.sum(err,axis=0)
    inliers = np.array(corrs)[dist<=delta]
    outliers = np.array(corrs)[dist>delta]
    return inliers, outliers

def RANSAC(corrs, epsilon, sigma, n, p):
    N = math.ceil(math.log(1-p)/math.log(1-(1-epsilon)**n))
    delta = 3*sigma
    n_total = len(corrs)
    M = math.ceil((1-epsilon)*n_total)
    outlier_cords = []
    inlier_cords = []
    numInliers = 0

    for iter in range(N):
        corrSample = random.sample(corrs, n)
        H_estimate = estimateH(corrSample)
        inliers, outliers = getInliers(H_estimate, corrs, delta)
        if len(inliers)>numInliers:
            numInliers = len(inliers)
            inlier_cords = inliers
            outlier_cords = outliers
    return inlier_cords, outlier_cords

def drawInliersOutliers(img0, img1, inliers, outliers, maxPts):
    w = img0.shape[1]
    comb_img1 = np.concatenate((img0, img1), axis=1)
    comb_img2 = comb_img1.copy()
    for i in range(len(inliers) if len(inliers)<maxPts else maxPts):
        pts = inliers[i]
        x = pts[0]
        xp = pts[1] + [w,0]
        cv.circle(comb_img1, tuple(x), 3, (0,255,0), 1)
        cv.circle(comb_img1, tuple(xp), 3, (0,255,0), 1)
        cv.line(comb_img1, tuple(x), tuple(xp), (0,255,0), 1)
    for i in range(len(outliers) if len(outliers)<maxPts else maxPts):
        pts = outliers[i]
        x = pts[0]
        xp = pts[1] + [w,0]
        cv.circle(comb_img2, tuple(x), 3, (0,0,255), 1)
        cv.circle(comb_img2, tuple(xp), 3, (0,0,255), 1)
        cv.line(comb_img2, tuple(x), tuple(xp), (0,0,255), 1)
    
    return comb_img1, comb_img2

def costFunc(h, inliers):
    X = []
    F = []
    for i in range(len(inliers)):
        pair = inliers[i]
        d_x = pair[0][0]
        d_y = pair[0][1]
        r_x = pair[1][0]
        r_y = pair[1][1]
        X.append(r_x)
        X.append(r_y)
        F.append((h[0]*d_x + h[1]*d_y + h[2])/(h[6]*d_x + h[7]*d_y + h[8])) 
        F.append((h[3]*d_x + h[4]*d_y + h[5])/(h[6]*d_x + h[7]*d_y + h[8]))
    err = np.array(X) - np.array(F)

    return err

def getPixelValue(img, coord):
    pt = np.array([coord[0], coord[1]])
    x = int(np.floor(coord[0]))
    y = int(np.floor(coord[1]))
    d1 = np.linalg.norm(pt - np.array([x,y]))
    d2 = np.linalg.norm(pt - np.array([x, y+1]))
    d3 = np.linalg.norm(pt - np.array([x+1, y]))
    d4 = np.linalg.norm(pt - np.array([x+1, y+1]))
    pixel = (img[y][x][:]*d1 + img[y+1][x][:]*d2 + img[y][x+1][:]*d3 + img[y+1][x+1][:]*d4)/(d1+d2+d3+d4)
    return pixel

def mapPixels(canvas, img, h):
    h = np.linalg.pinv(h)

    # Create a matrix of coordinates for the canvas
    worldCoords = np.array([[i, j, 1] for i in range(canvas.shape[1]) for j in range(canvas.shape[0])])
    worldCoords = np.transpose(worldCoords)
    # Transform those coordinates using H
    imgCoords = np.matmul(h, worldCoords)
    imgCoords = imgCoords/imgCoords[2,:]

    # isolate the pixel coordinates that will change based on the current image
    temp = imgCoords[0, :] >= 0
    worldCoords = worldCoords[:, temp]
    imgCoords = imgCoords[:, temp]

    temp = imgCoords[0, :] <= img.shape[1]-1
    worldCoords = worldCoords[:, temp]
    imgCoords = imgCoords[:, temp]

    temp = imgCoords[1, :] >= 0
    worldCoords = worldCoords[:, temp]
    imgCoords = imgCoords[:, temp]

    temp = imgCoords[1, :] <= img.shape[0]-1
    worldCoords = worldCoords[:, temp]
    imgCoords = imgCoords[:, temp]

    pts = worldCoords.shape[1]
    for i in range(pts):
        point = imgCoords[:,i]
        y = worldCoords[1][i]
        x = worldCoords[0][i]
        canvas[y][x][:] = getPixelValue(img, point)

    return canvas

def panorama(imgs, H):

    # Taking center image as reference image
    refImg = int((len(imgs)+1)/2)-1
    print("Ref Img: ",refImg)

    h_temp = np.eye(3)
    # Finding H to tranform each image to refImg frame 
    for i in range(refImg-1,-1,-1):
        h_temp = np.matmul(h_temp, H[i])
        H[i] = h_temp
    
    h_temp = np.eye(3)
    for i in range(refImg, len(H)):
        h_temp = np.matmul(h_temp, np.linalg.pinv(H[i]))
        H[i] = h_temp
    
    H.insert(refImg, np.eye(3, dtype=np.float64))

    # Add translation for refImg to be in the middle of panorama
    tx = 0
    for i in range(refImg):
        tx = tx + imgs[i].shape[1]
    H_t = np.array([1, 0, tx, 0, 1, 0, 0, 0, 1], dtype=float)
    H_t = H_t.reshape(3,3)
    for i in range(len(H)):
        H[i] = np.matmul(H_t, H[i])
    
    # Create the canvas
    canvas_w = 0
    canvas_h = 0
    for img in imgs:
        img_w = img.shape[1]
        img_h = img.shape[0]
        canvas_h = max(img_h, canvas_h)
        canvas_w = canvas_w + img_w
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    
    # Map pixels
    for i, img in enumerate(imgs):
        canvas = mapPixels(canvas, img, H[i])    

    return canvas
#-----------------------------Task 1------------------------------#

# HW images
img0 = cv.imread('HW5/0.jpg')
img1 = cv.imread('HW5/1.jpg')
img2 = cv.imread('HW5/2.jpg')
img3 = cv.imread('HW5/3.jpg')
img4 = cv.imread('HW5/4.jpg')

# My own Images
img5 = cv.imread('HW5/5.jpg')
img6 = cv.imread('HW5/6.jpg')
img7 = cv.imread('HW5/7.jpg')
img8 = cv.imread('HW5/8.jpg')
img9 = cv.imread('HW5/9.jpg')
img10 = cv.imread('HW5/10.jpg')

# imgs = [img0, img1, img2, img3, img4]

imgs = [img5, img6, img7, img8, img9, img10]

hMatrix_LS = []
hMatrix_LM = []

for i in range(len(imgs)-1):
    # Get images
    img0 = imgs[i]
    img1 = imgs[i+1]

    # Use SIFT to find key correpondences
    print("Computing SIFT Correspondences")
    corres = sift(img0, img1)
    corres_img = drawCorrespondences(corres, img0, img1, 250)
    cv.imwrite('HW5/img_'+str(i)+str(i+1)+'_corres_img.jpeg', corres_img)
    print(str(len(corres))+" SIFT Correspondences found between Image "+str(i)+" and Image "+str(i+1))

    # Use RANSAC to get inliers and outliers
    print("Applying RANSAC")
    inliers, outliers = RANSAC(corres, epsilon=0.75, sigma=2, n=6, p=0.999)
    inliers_img, outliers_img = drawInliersOutliers(img0,img1,inliers,outliers, 350)
    cv.imwrite('HW5/img_'+str(i)+str(i+1)+'_inliers_img.jpeg', inliers_img)
    cv.imwrite('HW5/img_'+str(i)+str(i+1)+'_outliers_img.jpeg', outliers_img)
    print("RANSAC completed. Inliers: "+str(len(inliers))+" Outliers: "+str(len(outliers)))

    # Get a better estimate of H from all the inliers
    H_estimate_LS = estimateH(inliers)
    hMatrix_LS.append(H_estimate_LS)

    # Using LM to get a more refined H estimate
    H_estimate_LM = H_estimate_LS.reshape((1,9))[0]
    H_estimate_LM = optimize.least_squares(costFunc, H_estimate_LM, args=[inliers], method='lm').x
    H_estimate_LM = H_estimate_LM.reshape((3,3))
    hMatrix_LM.append(H_estimate_LM)

# Build the Panaroma
print("Building Panorama")
img_panorama_LS = panorama(imgs, hMatrix_LS)
img_panorama_LM = panorama(imgs, hMatrix_LM)
cv.imwrite('HW5/img2_panorama_LS.jpeg', img_panorama_LS)
cv.imwrite('HW5/img2_panorama_LM.jpeg', img_panorama_LM)

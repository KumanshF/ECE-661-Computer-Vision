import numpy as np
import cv2 as cv
import math
from random import choice

#-------------------------Functions-------------------------------#
def HaarMat(sigma):
    M = math.ceil(4*sigma) + (math.ceil(4*sigma)%2==1)
    kx = np.ones((M,M))
    ky = np.ones((M,M))
    kx[:,:M//2] = -1
    ky[M//2:,:] = -1
    return kx,ky

def Harris(img_raw, sigma):
    if len(img_raw.shape) == 3:
        img = cv.cvtColor(img_raw,cv.COLOR_BGR2GRAY)
    else:
        img = img_raw
    img = img.astype(np.float64)
    img -= np.min(img)
    img /= np.max(img) 

    # Get kernel kx and ky to estimate dx and dy
    kx, ky = HaarMat(sigma)
    dx = cv.filter2D(img, -1, kernel = kx)
    dy = cv.filter2D(img, -1, kernel = ky)

    # Create the 5sigma neighborhood
    N = 2*int((5*sigma)//2) + 1
    k_N = np.ones((N,N))
    # Create the C matrix
    dx_dx = dx * dx
    dy_dy = dy * dy
    dx_dy = dx * dy

    s_dxdx = cv.filter2D(dx_dx, -1, kernel = k_N)
    s_dydy = cv.filter2D(dy_dy, -1, kernel = k_N)
    s_dxdy = cv.filter2D(dx_dy, -1, kernel = k_N)
    
    # find det(C) and Tr(C)
    det = s_dxdx*s_dydy - (s_dxdy*s_dxdy)
    tr = s_dxdx + s_dydy

    k_vals = det / (tr**2 + 0.001)
    k = np.sum(k_vals) / (img.shape[0]*img.shape[1])

    R = det - k * (tr**2)
    thres = np.sort(R, axis=None)[-int(0.05*len(R.flatten()))]

    # Non-maximum suppression to remove nonrelevant coords
    r = int(N/2)
    coords = []
    for y in range(r, img.shape[0]-r):
        for x in range(r, img.shape[1]-r):
            roi = R[y-r : y+r , x-r : x+r]
            if R[y,x] == np.amax(roi) and np.amax(roi) >= thres and R[y,x]>0:
                coords.append([x,y])
    return coords

def drawCircles(img, corners):
    img_copy = img.copy()
    for corner in corners:
        x,y = corner
        cv.circle(img_copy, (x,y), 2, (10,240,10), -1)
    return img_copy

def getDistance(img1, img2, coord1, coord2, metric):
    M = math.ceil(4*sigma) + (math.ceil(4*sigma)%2==1)
    x1,y1 = coord1
    x2,y2 = coord2

    r = min(x1, y1, img1.shape[1] - x1, img1.shape[0]-y1, x2, y2, img2.shape[1] - x2, img2.shape[0]-y2, M+1)
    reg1 = img1[y1-r : y1+r, x1-r : x1+r]
    reg2 = img2[y2-r : y2+r, x2-r : x2+r]
    
    if metric == 'SSD':
        d = np.sum(np.square(reg1-reg2))
    elif metric == 'NCC':
        m1 = np.mean(reg1)
        m2 = np.mean(reg2)
        d = 1 - np.sum((reg1-m1)*(reg2-m2))/(np.sqrt(np.sum(np.square(reg1-m1))*np.sum(np.square(reg2-m2))))
    return d

def getCorresp(img1_raw, img2_raw, corner_coords1, corner_coords2, sigma, metric):
    if len(img1_raw.shape) == 3:
        img1 = cv.cvtColor(img1_raw,cv.COLOR_BGR2GRAY)
    else:
        img1 = img1_raw
    img1 = img1.astype(np.float64)
    img1 -= np.min(img1)
    img1 /= np.max(img1)

    if len(img2_raw.shape) == 3:
        img2 = cv.cvtColor(img2_raw,cv.COLOR_BGR2GRAY)
    else:
        img2 = img2_raw
    img2 = img2.astype(np.float64)
    img2 -= np.min(img2)
    img2 /= np.max(img2)
    
    corrs = []
    if len(corner_coords1) <= len(corner_coords2):
        for coord1 in corner_coords1:
            bestP = 0
            bestD = float('inf')
            for coord2 in corner_coords2:
                d = getDistance(img1, img2, coord1, coord2, metric)
                if d < bestD:
                    bestD = d
                    bestP = coord2
            corrs.append((coord1, bestP, bestD))
    else: 
        for coord2 in corner_coords2:
            bestP = 0
            bestD = float('inf')
            for coord1 in corner_coords1:
                d = getDistance(img1, img2, coord1, coord2, metric)
                if d < bestD:
                    bestD = d
                    bestP = coord1
            corrs.append((bestP, coord2, bestD))
    
    return corrs

def drawCorrs(img1_raw, img2_raw, corrs, N):
    # resize images: 
    h1 = img1_raw.shape[0]
    w1 = img1_raw.shape[1]
    h2 = img2_raw.shape[0]
    w2 = img2_raw.shape[1]
    if h1>h2:
        img1_raw = cv.resize(img1_raw, (w2,h2), cv.INTER_AREA)
    else:
        img2_raw = cv.resize(img2_raw, (w1,h1), cv.INTER_AREA)

    # Join the images next to each other to show correspondence
    comb_img = np.concatenate((img1_raw, img2_raw), axis=1)
    width = img1_raw.shape[1]

    # Show N first correspondences
    if N == 0:
        N = len(corrs)
    for i in range(N):
        x1,y1 = corrs[i][0]
        x2,y2 = corrs[i][1]
        x2 = x2 + width        
        cv.circle(comb_img, (x1,y1), 3, (10, 240, 10), -1)
        cv.circle(comb_img, (x2,y2), 3, (240, 240,10), -1)
        cv.line(comb_img, (x1,y1), (x2,y2), (10, 240,240), 1)
    return comb_img

def sift(img1_raw, img2_raw):
    if len(img1_raw.shape) == 3:
        img1 = cv.cvtColor(img1_raw,cv.COLOR_BGR2GRAY)
    else:
        img1 = img1_raw

    if len(img2_raw.shape) == 3:
        img2 = cv.cvtColor(img2_raw,cv.COLOR_BGR2GRAY)
    else:
        img2 = img2_raw

    corrs = []

    # Create sift detector
    sift_detector = cv.SIFT_create()

    # find key points and descriptors with Sift
    kp1, des1 = sift_detector.detectAndCompute(img1,None)
    kp2, des2 = sift_detector.detectAndCompute(img2,None)

    # BFMatcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    corrs = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            corrs.append([m])
    comb_img = np.concatenate((img1_raw, img2_raw), axis=1)
    cv.drawMatchesKnn(img1_raw, kp1, img2_raw, kp2, corrs[0:150], comb_img,flags=2)

    return comb_img


#----------------------------------Inputs---------------------------------#

# Input Images
book1_img = cv.imread('HW4/books_1.jpeg')
book2_img = cv.imread('HW4/books_2.jpeg')
fountain1_img = cv.imread('HW4/fountain_1.jpg')
fountain2_img = cv.imread('HW4/fountain_2.jpg')
lounge1_img = cv.imread('HW4/lounge1.jpg')
lounge2_img = cv.imread('HW4/lounge2.jpg')
studio1_img = cv.imread('HW4/studio1.jpg')
studio2_img = cv.imread('HW4/studio2.jpg')


# Input scale sigma
sigmas = [0.8, 1.2, 1.4, 1.6]

#--------------------Task 1.1: Harris Corner Detection--------------------#

for sigma in sigmas:
    print("Sigma:", sigma)
    
    #----------------1st Pair--------------#
    # Find Interest Points
    book1_corners = Harris(book1_img, sigma)
    print("Interest points for books1:",len(book1_corners))
    book1_corners_img = drawCircles(book1_img, book1_corners)

    book2_corners = Harris(book2_img, sigma)
    print("Interest points for books2:",len(book2_corners))
    book2_corners_img = drawCircles(book2_img, book2_corners)

    # Get Correspondence using SSD
    book_corrs_SSD = getCorresp(book1_img, book2_img, book1_corners, book2_corners, sigma, 'SSD')
    book_corrs_SSD_img = drawCorrs(book1_img, book2_img, book_corrs_SSD, 100)

    # Get Correspondence using NCC
    book_corrs_NCC = getCorresp(book1_img, book2_img, book1_corners, book2_corners, sigma, 'NCC')
    book_corrs_NCC_img = drawCorrs(book1_img, book2_img, book_corrs_NCC, 100)
    
    # Save images
    cv.imwrite('HW4/book1_'+str(sigma)+'.jpeg', book1_corners_img)
    cv.imwrite('HW4/book2_'+str(sigma)+'.jpeg', book2_corners_img)
    cv.imwrite('HW4/book_corrs_SSD'+str(sigma)+'.jpeg', book_corrs_SSD_img)
    cv.imwrite('HW4/book_corrs_NCC'+str(sigma)+'.jpeg', book_corrs_NCC_img)
    
    #----------------2nd Pair--------------#
    # Find Interest Points
    fountain1_corners = Harris(fountain1_img, sigma)
    print("Interest Points for Fountain:",len(fountain1_corners))
    fountain1_corners_img = drawCircles(fountain1_img, fountain1_corners)

    fountain2_corners = Harris(fountain2_img, sigma)
    print("Interest Points for Fountain2:",len(fountain2_corners))
    fountain2_corners_img = drawCircles(fountain2_img, fountain2_corners)
    
    # Get Correspondence using SSD
    fountain_corrs_SSD = getCorresp(fountain1_img, fountain2_img, fountain1_corners, fountain2_corners, sigma, 'SSD')
    fountain_corrs_SSD_img = drawCorrs(fountain1_img, fountain2_img, fountain_corrs_SSD, 100)

    # Get Correspondence using NCC
    fountain_corrs_NCC = getCorresp(fountain1_img, fountain2_img, fountain1_corners, fountain2_corners, sigma, 'NCC')
    fountain_corrs_NCC_img = drawCorrs(fountain1_img, fountain2_img, fountain_corrs_NCC, 100)    

    # Save Images
    cv.imwrite('HW4/fountain1_'+str(sigma)+'.jpeg', fountain1_corners_img)
    cv.imwrite('HW4/fountain2_'+str(sigma)+'.jpeg', fountain2_corners_img)
    cv.imwrite('HW4/fountain_corrs_SSD'+str(sigma)+'.jpeg', fountain_corrs_SSD_img)
    cv.imwrite('HW4/fountain_corrs_NCC'+str(sigma)+'.jpeg', fountain_corrs_NCC_img)
    
    #----------------3rd Pair--------------#
    # Find Interest Points
    lounge1_corners = Harris(lounge1_img, sigma)
    print("Interest points for lounge1:",len(lounge1_corners))
    lounge1_corners_img = drawCircles(lounge1_img, lounge1_corners)

    lounge2_corners = Harris(lounge2_img, sigma)
    print("Interest points for lounge2:",len(lounge2_corners))
    lounge2_corners_img = drawCircles(lounge2_img, lounge2_corners)

    # Get Correspondence using SSD
    lounge_corrs_SSD = getCorresp(lounge1_img, lounge2_img, lounge1_corners, lounge2_corners, sigma, 'SSD')
    lounge_corrs_SSD_img = drawCorrs(lounge1_img, lounge2_img, lounge_corrs_SSD, 100)

    # Get Correspondence using NCC
    lounge_corrs_NCC = getCorresp(lounge1_img, lounge2_img, lounge1_corners, lounge2_corners, sigma, 'NCC')
    lounge_corrs_NCC_img = drawCorrs(lounge1_img, lounge2_img, lounge_corrs_NCC, 100)
    
    # Save images
    cv.imwrite('HW4/lounge1_'+str(sigma)+'.jpeg', lounge1_corners_img)
    cv.imwrite('HW4/lounge2_'+str(sigma)+'.jpeg', lounge2_corners_img)
    cv.imwrite('HW4/lounge_corrs_SSD'+str(sigma)+'.jpeg', lounge_corrs_SSD_img)
    cv.imwrite('HW4/lounge_corrs_NCC'+str(sigma)+'.jpeg', lounge_corrs_NCC_img)

    #----------------4th Pair--------------#
    # Find Interest Points
    studio1_corners = Harris(studio1_img, sigma)
    print("Interest points for studio1:",len(studio1_corners))
    studio1_corners_img = drawCircles(studio1_img, studio1_corners)

    studio2_corners = Harris(studio2_img, sigma)
    print("Interest points for studio2:",len(studio2_corners))
    studio2_corners_img = drawCircles(studio2_img, studio2_corners)

    # Get Correspondence using SSD
    studio_corrs_SSD = getCorresp(studio1_img, studio2_img, studio1_corners, studio2_corners, sigma, 'SSD')
    studio_corrs_SSD_img = drawCorrs(studio1_img, studio2_img, studio_corrs_SSD, 0)

    # Get Correspondence using NCC
    studio_corrs_NCC = getCorresp(studio1_img, studio2_img, studio1_corners, studio2_corners, sigma, 'NCC')
    studio_corrs_NCC_img = drawCorrs(studio1_img, studio2_img, studio_corrs_NCC, 0)
    
    # Save images
    cv.imwrite('HW4/studio1_'+str(sigma)+'.jpeg', studio1_corners_img)
    cv.imwrite('HW4/studio2_'+str(sigma)+'.jpeg', studio2_corners_img)
    cv.imwrite('HW4/studio_corrs_SSD'+str(sigma)+'.jpeg', studio_corrs_SSD_img)
    cv.imwrite('HW4/studio_corrs_NCC'+str(sigma)+'.jpeg', studio_corrs_NCC_img)

# Get Interest points using SIFT
book_corrs_sift_img = sift(book1_img, book2_img)
cv.imwrite('HW4/book_corrs_Sift.jpeg', book_corrs_sift_img)

# Get Interest points using SIFT
fountain_corrs_sift_img = sift(fountain1_img, fountain2_img)
cv.imwrite('HW4/fountain_corrs_sift.jpeg', fountain_corrs_sift_img)

# Get Interest points using SIFT
lounge_corrs_sift_img = sift(lounge1_img, lounge2_img)
cv.imwrite('HW4/lounge_corrs_sift.jpeg', lounge_corrs_sift_img)

# Get Interest points using SIFT
studio_corrs_sift_img = sift(studio1_img, studio2_img)
cv.imwrite('HW4/studio_corrs_sift.jpeg', studio_corrs_sift_img)

# cv.imshow('1',img_01_corres_img)
 
# cv.waitKey(0)
# cv.destroyAllWindows()
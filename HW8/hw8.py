import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import scipy.optimize
#========================Functions========================#
#### Corner Detection Functions
def getFixedPts(gridSize, hlines, vlines):
    x = np.linspace(0, gridSize*(vlines-1), vlines)
    y = np.linspace(0, gridSize*(hlines-1), hlines)
    xmesh, ymesh = np.meshgrid(x,y)
    fixedPts = np.concatenate([xmesh.reshape((-1,1)), ymesh.reshape((-1,1))], axis=1)
    return fixedPts

def getPoints(lines):
    '''
    Code referenced from https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    '''
    rho = lines[:,0]
    theta = lines[:,1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    p1 = np.array([x0 + 1000*(-b), y0 + 1000*(a)])
    p2 = np.array([x0 - 1000*(-b), y0 - 1000*(a)])
    return p1.T, p2.T

def drawLines(image, pts1, pts2, color, t=1):
    img = image.copy()
    for i in range(0, pts1.shape[0]):
        pt1 = (int(pts1[i,0]),int(pts1[i,1]))
        pt2 = (int(pts2[i,0]),int(pts2[i,1]))
        cv.line(img, pt1, pt2, color, t)
    return img

def drawPoints(image, pts, r, color, t=1, text = False):
    img = image.copy()
    for i in range(pts.shape[0]):
        pt1 = int(pts[i,0])
        pt2 = int(pts[i,1])
        cv.circle(img, (pt1,pt2), r, color, -1)
        if text:
            cv.putText(img, str(i+1), (pt1+5,pt2+5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
    return img

def getLines(imgName, num, r, thres, dataset):
    img = cv.imread(imgName)
    # Canny and Hough Transform to get lines
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if dataset==2:
        gray = cv.GaussianBlur(gray,(3,3),2)
    edges = cv.Canny(gray, 255*1.5, 255)
    lines = cv.HoughLines(edges, 1, r*np.pi/180, thres)
    lines = np.squeeze(lines)
    if dataset ==1:
        if num<5:
            # Save Canny Image
            cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_canny_d1.jpg', edges)
            # Save HoughLines Image
            p1, p2 = getPoints(lines)
            img_HoughLines = drawLines(img, p1, p2, (255,255,10), 1)
            cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_HoughL_d1.jpg', img_HoughLines)
    if dataset==2:
        if num<12:
            # Save Canny Image
            cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_canny_d2.jpg', edges)
            # Save HoughLines Image
            p1, p2 = getPoints(lines)
            img_HoughLines = drawLines(img, p1, p2, (255,255,10), 1)
            cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_HoughL_d2.jpg', img_HoughLines) 
    return lines

def refineLines(lines_sorted, nms_r, v=0):
    validLines = []
    if v == 1:# if vertical
        d = lines_sorted[:,0]*np.cos(lines_sorted[:,1]) # get distance (already sorted from before)
        d_thres = nms_r * (np.max(np.abs(d))-np.min(np.abs(d)))/7 # average distance ratio
    else:
        d = lines_sorted[:,0]*np.sin(lines_sorted[:,1])
        d_thres = nms_r * (np.max(np.abs(d))-np.min(np.abs(d)))/9
    for i in range(d.shape[0]-1):
        if d[i+1] - d[i] > d_thres:
            #if the distance between lines is > thresh, take first line 
            validLines.append([lines_sorted[i,0], lines_sorted[i,1]])
        if i==d.shape[0]-2:
            validLines.append([lines_sorted[i+1,0], lines_sorted[i+1,1]])
    return np.asarray(validLines)

def getCorners(imgName, num, lines, nms, dataset):
    img = cv.imread(imgName)
    # get vertical and horizontal lines
    vlines = lines[np.where(np.cos(lines[:,1])**2 > 0.5)]
    hlines = lines[np.where(np.cos(lines[:,1])**2 <=0.5)]

    # Sort the lines (hlines-> up to down, vlines -> left to right)
    vlines_sort = vlines[:,0]*np.cos(vlines[:,1])
    hlines_sort = hlines[:,0]*np.sin(hlines[:,1])
    vlines_sorted = vlines[np.argsort(vlines_sort)]
    hlines_sorted = hlines[np.argsort(hlines_sort)]

    # Refine lines
    vlines_refined = refineLines(vlines_sorted, nms, v = 1)
    hlines_refined = refineLines(hlines_sorted, nms)

    vpt1, vpt2 = getPoints(vlines_refined)
    hpt1, hpt2 = getPoints(hlines_refined)
    imgLines = drawLines(img, vpt1, vpt2, (10,10,255), 1)
    imgLines = drawLines(imgLines, hpt1, hpt2, (10,10,255), 1)

    # Save refined lines image
    if dataset==1:
        cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_lines_d1.jpg', imgLines)
    elif dataset==2:
        cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_lines_d2.jpg', imgLines)

    # Get corners
    vpt1_hc = np.append(vpt1, np.ones((vpt1.shape[0],1)), axis=1)
    vpt2_hc = np.append(vpt2, np.ones((vpt2.shape[0],1)), axis=1)
    hpt1_hc = np.append(hpt1, np.ones((hpt1.shape[0],1)), axis=1)
    hpt2_hc = np.append(hpt2, np.ones((hpt2.shape[0],1)), axis=1)
    vlines_HC = np.cross(vpt1_hc, vpt2_hc)  #valid vertical lines
    hlines_HC = np.cross(hpt1_hc, hpt2_hc)  #valid horizontal lines

    corners = []
    for i in range(hlines_HC.shape[0]):
        # Intersection of 2 lines
        c = np.cross(vlines_HC, hlines_HC[i])
        c = c[:,:2]/c[:,2].reshape((-1,1))
        corners.append(c)
    corners = np.asarray(corners)
    corners = corners.reshape((-1,2))
    imgPoints = drawPoints(img, corners, 2, (255,90,10), 1, text=True)
    if dataset==1:
        cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_points_d1.jpg', imgPoints)
    elif dataset==2:
        cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/CornersAndLines/Pics_'+str(num+1)+'_points_d2.jpg', imgPoints)
    return corners

def getHomography(domainPts, RangePts):
    # rangePts = H*domainPts
    numPts = len(domainPts)
     # A and B Matrix
    A = np.zeros((2*numPts,8))
    b = np.zeros((2*numPts,1))
    for i in range(numPts):
        xp = RangePts[i,0]
        yp = RangePts[i,1]
        x = domainPts[i,0]
        y = domainPts[i,1]
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

#### Camera Calibration Functions

def calcVij(hi,hj):
    Vij = np.array([hi[0]*hj[0], 
                    hi[0]*hj[1] + hi[1]*hj[0], 
                    hi[1]*hj[1], 
                    hi[2]*hj[0] + hi[0]*hj[2], 
                    hi[2]*hj[1] + hi[1]*hj[2], 
                    hi[2]*hj[2]])
    return Vij.T

def getOmega(Hs):
    # vb = 0
    V = []
    for H in Hs:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        v11 = calcVij(h1,h1)
        v12 = calcVij(h1,h2)
        v22 = calcVij(h2,h2)
        V.append(v12.T)
        V.append((v11-v22).T)
    
    # Compute SVD of v to find b
    u,s,v = np.linalg.svd(V)
    b = v[-1]
    omega = np.zeros((3,3))
    omega[0,0] = b[0]
    omega[0,1] = b[1]
    omega[0,2] = b[3]
    omega[1,0] = b[1]
    omega[1,1] = b[2]
    omega[1,2] = b[4]
    omega[2,0] = b[3]
    omega[2,1] = b[4]
    omega[2,2] = b[5]
    return omega

def getK(w):
    y_0 = (w[0,1] * w[0,2] - w[0,0] * w[1,2])/(w[0,0] * w[1,1] - w[0,1]**2)
    lamb = w[2,2] - (w[0,2]**2 + y_0 * (w[0,1] * w[0,2] - w[0,0] * w[1,2]))/w[0,0]
    alpha_x = np.sqrt(lamb/w[0,0])
    alpha_y = np.sqrt(lamb*(w[0,0]/(w[0,0] * w[1,1] - w[0,1]**2)))
    s = -w[0,1] * alpha_x**2 * alpha_y/ lamb
    x_0 = (s * y_0 / alpha_y) - (w[0,2] * alpha_x**2 / lamb)
    K = np.array([[alpha_x, s, x_0], [0, alpha_y, y_0], [0, 0, 1]])
    return K

def getRt(Hs, K):
    RList = []
    tList = []
    for H in Hs:
        rtMat = np.dot(np.linalg.inv(K), H)
        scale = 1/(np.linalg.norm(rtMat[:,0]))
        rtMat = scale*rtMat
        r1 = rtMat[:,0]
        r2 = rtMat[:,1]
        r3 = np.cross(r1,r2)
        Q = np.vstack((r1,r2, r3)).T
        u,_,v = np.linalg.svd(Q)
        R = np.matmul(u,v)
        RList.append(R)
        tList.append(rtMat[:,2])    
    return RList, tList

def getParams(K, RList, tList):
    # K params
    alpha_x = K[0,0]
    s = K[0,1]
    x_0 = K[0,2]
    alpha_y = K[1,1]
    y_0 = K[1,2]
    K_param = np.asarray([alpha_x, s, x_0, alpha_y, y_0])
    
    # Convert R into 3DOF
    RtList_i = []
    for R,t in zip(RList, tList):
        val = (np.trace(R)-1)/2
        phi = np.arccos(val)
        W = phi / (2*np.sin(phi)) * np.asarray([R[2,1]-R[1,2], 
                                                R[0,2]-R[2,0],
                                                R[1,0]-R[0,1]])
        Rt_i = np.append(W,t)
        RtList_i.append(Rt_i)
    p = np.append(K_param, np.concatenate(RtList_i))
    return p

def getKRt_mat(params):
    numRt_mats = int((params.shape[0]-5) / 6)
    Ks = params[:5]
    K = np.array([[Ks[0], Ks[1], Ks[2]], [0, Ks[3], Ks[4]], [0, 0, 1]])

    RList = [] 
    tList = []
    Rts = params[5:]
    for i in range(numRt_mats):
        W = Rts[i*6:i*6+3]
        t = Rts[i*6+3:i*6+6]
        phi = np.linalg.norm(W)
        Wx = np.array([[0, -W[2], W[1]], [W[2], 0, -W[0]], [-W[1], W[0], 0]])
        R = np.eye(3) + (np.sin(phi)/phi)*Wx + ((1-np.cos(phi))/(phi**2))*np.matmul(Wx,Wx)
        RList.append(R)
        tList.append(t)
    return K, RList, tList

def projPoints(H, coords):
    coords_hc = np.concatenate((coords, np.ones((coords.shape[0],1))), axis=1)
    projectedCoords_hc = np.dot(H, coords_hc.T).T 
    projectedPts = projectedCoords_hc[:,:2]/projectedCoords_hc[:,2].reshape((coords.shape[0],1))
    return projectedPts

def lossFunc(params, corners, fixedPoints):
    ### d_geom = ||Xij - X^ij||
    K, RList, tList = getKRt_mat(params)
    projected_corners = []
    for R, t in zip(RList,tList):
        rt = np.vstack((R[:,0], R[:,1], t)).T
        H = np.dot(K, rt)
        projectedPts = projPoints(H, fixedPoints)
        projected_corners.append(projectedPts)
    
    projected_corners = np.concatenate(projected_corners, axis=0)
    actual_corners = np.concatenate(corners, axis=0)
    diff = projected_corners - actual_corners
    d_geom = diff.flatten()
    return d_geom

def reprojectGT(params, corners, fixedPoints, imgIdx, imgList, dataset, lm=False):
    K, RList, tList = getKRt_mat(params)
    Hs = []
    for R,t in zip(RList, tList):
        rt = np.vstack((R[:,0], R[:,1], t)).T
        H = np.dot(K, rt)
        Hs.append(H)
    
    diff_corners = []
    for i, H in enumerate(Hs):
        imageName = imgList[imgIdx[i]]
        img = cv.imread(imageName)    
        projectedPts = projPoints(H, fixedPoints)
        imgPoints = drawPoints(img, projectedPts, 2, (255,90,10), 1, text = True)
        if lm == False:
            imgPoints = drawPoints(imgPoints, projectedPts, 2, (10,255,10), 1, text = False)
        else: 
            imgPoints = drawPoints(imgPoints, projectedPts, 2, (10,10,255), 1, text = False)
        if dataset==1:
            if lm==False:
                cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/beforelm/Pics_'+str(imgIdx[i]+1)+'_blm_d1.jpg', imgPoints)
            else: 
                cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/afterlm/Pics_'+str(imgIdx[i]+1)+'_alm_d1.jpg', imgPoints)
        elif dataset==2:
            if lm==False:
                cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/beforelm/Pics_'+str(imgIdx[i]+1)+'_blm_d2.jpg', imgPoints)
            else: 
                cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/afterlm/Pics_'+str(imgIdx[i]+1)+'_alm_d2.jpg', imgPoints)
        diff = projectedPts - corners[i]
        diff_corners.append(diff)
    
    diff_corners = np.array(diff_corners)
    diff_corners = diff_corners.reshape((-1,2))
    diff_indNorm = np.linalg.norm(diff_corners, axis=1)
    # Get individual mean, var and max
    avg = []
    var = []
    maxd = []
    for i in range(int(diff_indNorm.shape[0]/80)):
        cset = diff_indNorm[i*80 : i*80+80]
        avg.append(np.average(cset))
        var.append(np.var(cset))
        maxd.append(np.max(cset))
  
    # Get overall accuracy
    avg_all = np.average(diff_indNorm)
    var_all = np.var(diff_indNorm)
    max_d_all = np.max(diff_indNorm)

    return np.array([avg, var, maxd]), np.array([avg_all, var_all, max_d_all])

def reprojectCorners(params, corners, imgIdx, imgList, fixedImgIdx, dataset, lm = False):
    K, RList, tList = getKRt_mat(params)
    Hs = []
    for R,t in zip(RList, tList):
        rt = np.vstack((R[:,0], R[:,1], t)).T
        H = np.dot(K, rt)
        Hs.append(H)

    # Index for the fixed image changes based on the parameters used
    fixedIdx = np.where(imgIdx==fixedImgIdx)
    fixedIdx = fixedIdx[0][0]-1
    fixedImg_H = Hs[fixedIdx] # img 28
    fixedCorners = corners[fixedIdx]
    fixedImg = cv.imread(imgList[fixedImgIdx-1])
    fixedImg = drawPoints(fixedImg, fixedCorners, 2, (255,90,10),1, text = True)

    diff_corners = []   
    for i, H in enumerate(Hs):
        validIdx = imgIdx[i]
        if validIdx == fixedIdx:
            continue
        H_bet = np.dot(fixedImg_H, np.linalg.inv(H))
        projectedPts = projPoints(H_bet, corners[i])
        if lm == False:
            imgPoints = drawPoints(fixedImg, projectedPts, 2, (10,255,10), 1, text = False)
        else: 
            imgPoints = drawPoints(fixedImg, projectedPts, 2, (10,10,255), 1, text = False)
        if validIdx in (1,5,9,10,12,29):
            if dataset==1:
                if lm == False:
                    cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/reprojection/Pics_'+str(validIdx+1)+'_d1.jpg', imgPoints)
                else: 
                    cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/reprojection/Pics_'+str(validIdx+1)+'_lm_d1.jpg', imgPoints)
            elif dataset==2:
                if lm == False:
                    cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/reprojection/Pics_'+str(validIdx+1)+'_d2.jpg', imgPoints)
                else: 
                    cv.imwrite('HW8/outputs/dataset'+str(dataset)+'/reprojection/Pics_'+str(validIdx+1)+'_lm_d2.jpg', imgPoints)
        diff = projectedPts - fixedCorners
        diff_corners.append(diff)
    diff_corners = np.array(diff_corners)
    diff_corners = diff_corners.reshape((-1,2))
    diff_indNorm = np.linalg.norm(diff_corners, axis=1)
    avg = []
    var = []
    maxd = []
    for i in range(int(diff_indNorm.shape[0]/80)):
        cset = diff_indNorm[i*80 : i*80+80]
        avg.append(np.average(cset))
        var.append(np.var(cset))
        maxd.append(np.max(cset))
    return np.array([avg, var, maxd])
#=========================Main Function==========================#
dataset = 2
## Get images
if dataset == 1:
    imageList = sorted(glob.glob('HW8/Dataset1/Pic_??.jpg'))
elif dataset == 2:
    imageList = sorted(glob.glob('HW8/Dataset2/Pic_??.jpg'))

fixedPts = getFixedPts(gridSize = 10, hlines = 10, vlines = 8)

## Parameters for datasets
# Dataset 1: r = 0.5, thres = 60, nms = 0.24
# Dataset 2: r = 0.5, thres = 50, nms = 0.24

### Step 1: Corner Detection and homography calculation ###
Hs = []
cornersList = []
imgIdx = []
for num, imgName in enumerate(imageList):
    # Get Lines using Canny and HoughLines
    lines = getLines(imgName, num, 0.5, 50, dataset)
    corners = getCorners(imgName, num, lines, 0.24, dataset)
    # Get Homography from the corners
    if corners.shape[0]==80:
        H = getHomography(fixedPts, corners)
        Hs.append(H)
        cornersList.append(corners)
        imgIdx.append(num)
print("Images Used:", len(cornersList))
imgIdx = np.array(imgIdx)
Hs = np.array(Hs)
cornersList = np.array(cornersList)

### Step 2: Calculate Intrinsic and Extrinsic Parameters ###
# Calculate Omega  
w = getOmega(Hs)
# Get K, [R|t] matrix for each image
K = getK(w)
RList, tList = getRt(Hs, K)
print("K before lm: \n",K)
if dataset==1:
    print("R before lm: \n",RList[3])
    print("t before lm: \n",tList[3])
    print("R before lm: \n",RList[4])
    print("t before lm: \n",tList[4])
elif dataset==2:
    print("R before lm: \n",RList[1])
    print("t before lm: \n",tList[1])
    print("R before lm: \n",RList[9])
    print("t before lm: \n",tList[9])
# Get initial values of the parameters p = [K, R_i, t_i | i=1,2,3...]^T
p = getParams(K, RList, tList)
# Reproject using parameters found before LM
acc, acc_all = reprojectGT(p, cornersList, fixedPts, imgIdx, imageList, dataset)

### Step 3: Refining the Calibration Parameters with LM ###
# Calculate loss
d_geom = lossFunc(p, cornersList, fixedPts)

# Apply LM
res = scipy.optimize.least_squares(fun=lossFunc, x0=p, method='lm', args = [cornersList, fixedPts])
K_refined, RList_refined, tList_refined = getKRt_mat(res.x)
print("K after lm: \n",K_refined)
if dataset==1:
    print("R after lm: \n",RList_refined[3])
    print("t after lm: \n",tList_refined[3])
    print("R after lm: \n",RList_refined[4])
    print("t after lm: \n",tList_refined[4])
elif dataset==2:
    print("R after lm: \n",RList_refined[1])
    print("t after lm: \n",tList_refined[1])
    print("R after lm: \n",RList_refined[9])
    print("t after lm: \n",tList_refined[9])
acc_lm, acc_lm_all = reprojectGT(res.x, cornersList, fixedPts, imgIdx, imageList, dataset, lm = True)

if dataset==1:
    # Overall Accuracy
    print("Overall Accuracy:")
    print(f'Before LM: Mean {acc_all[0]}, Var {acc_all[1]}, Max {acc_all[2]}')
    print(f'After LM: Mean {acc_lm_all[0]}, Var {acc_lm_all[1]}, Max {acc_lm_all[2]}\n')

    # Individual Accuracy
    print("Individual Accuracy")
    print("Before LM:")
    print(f'For Image 4: Mean {acc[0][3]}, Var {acc[1][3]}, Max {acc[2][3]}')
    print(f'For Image 5: Mean {acc[0][4]}, Var {acc[1][4]}, Max {acc[2][4]}')
    print("After LM:")
    print(f'For Image 4: Mean {acc_lm[0][3]}, Var {acc_lm[1][3]}, Max {acc_lm[2][3]}')
    print(f'For Image 5: Mean {acc_lm[0][4]}, Var {acc_lm[1][4]}, Max {acc_lm[2][4]}')

if dataset==2:
    # Overall Accuracy
    print("Overall Accuracy:")
    print(f'Before LM: Mean {acc_all[0]}, Var {acc_all[1]}, Max {acc_all[2]}')
    print(f'After LM: Mean {acc_lm_all[0]}, Var {acc_lm_all[1]}, Max {acc_lm_all[2]}\n')
    # Individual Accuracy
    print("Individual Accuracy")
    print("Before LM:")
    print(f'For Image 2: Mean {acc[0][1]}, Var {acc[1][1]}, Max {acc[2][1]}')
    print(f'For Image 10: Mean {acc[0][9]}, Var {acc[1][9]}, Max {acc[2][9]}')
    print("After LM:")
    print(f'For Image 2: Mean {acc_lm[0][1]}, Var {acc_lm[1][1]}, Max {acc_lm[2][1]}')
    print(f'For Image 10: Mean {acc_lm[0][9]}, Var {acc_lm[1][9]}, Max {acc_lm[2][9]}')

### Step 4: Reproject corners onto fixed image using the refined parameters
# Image 28 works well as a "fixed" image for dataset 1
# Image 10 works well as "fixed" image for dataset 2
if dataset==1:
    fixedImgIdx = 28
if dataset==2:
    fixedImgIdx = 10

acc_c = reprojectCorners(p, cornersList, imgIdx, imageList, fixedImgIdx, dataset)
acc_lm_c = reprojectCorners(res.x, cornersList, imgIdx, imageList, fixedImgIdx, dataset, lm = True)

# Print acc for three specific images (2,6)
print("Reprojection Accuracy")
print("Before LM:")
print(f'For Image 2: Mean {acc_c[0][1]}, Var {acc_c[1][1]}, Max {acc_c[2][1]}')
print(f'For Image 6: Mean {acc_c[0][5]}, Var {acc_c[1][5]}, Max {acc_c[2][5]}')
print("After LM:")
print(f'For Image 2: Mean {acc_lm_c[0][1]}, Var {acc_lm_c[1][1]}, Max {acc_lm_c[2][1]}')
print(f'For Image 6: Mean {acc_lm_c[0][5]}, Var {acc_lm_c[1][5]}, Max {acc_lm_c[2][5]}')
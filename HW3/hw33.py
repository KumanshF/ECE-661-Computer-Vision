import numpy as np
import cv2

#-------------------------Functions-------------------------------#
def drawCircleNBox(img, coords, coords2):
    img_box = img.copy()
    #draw circles at the coords
    for x,y in coords:
        cv2.circle(img_box, (x,y), 7, (255, 0, 0), -1)
    for x,y in coords2:
        cv2.circle(img_box, (x,y), 7, (255, 0, 0), -1)

    #draw lines connecting the coords to form a box
    coords.append(coords[0])
    for i in range(4):
        cv2.line(img_box, coords[i], coords[i+1], (0, 255, 0), 4)
    
    coords2.append(coords2[0])
    for i in range(4):
        cv2.line(img_box, coords2[i], coords2[i+1], (255, 255, 0), 4)

    #return Img
    return img_box

def getH(coords, coords2):
    p1 = (coords[0][0],coords[0][1],1)
    q1 = (coords[1][0],coords[1][1],1)
    r1 = (coords[2][0],coords[2][1],1)
    s1 = (coords[3][0],coords[3][1],1)

    p2 = (coords2[0][0],coords2[0][1],1)
    q2 = (coords2[1][0],coords2[1][1],1)
    r2 = (coords2[2][0],coords2[2][1],1)
    s2 = (coords2[3][0],coords2[3][1],1)

    l1 = np.cross(p1,q1)
    l2 = np.cross(p1,s1)
    l1 = l1/l1[2]
    l2 = l2/l2[2]

    l3 = np.cross(p1,s1)
    l4 = np.cross(s1,r1)
    l3 = l3/l3[2]
    l4 = l4/l4[2]

    l5 = np.cross(s1,r1)
    l6 = np.cross(r1,q1)
    l5 = l5/l5[2]
    l6 = l6/l6[2]

    l7 = np.cross(r1,q1)
    l8 = np.cross(q1,p1)
    l7 = l7/l7[2]
    l8 = l8/l8[2]

    l9 = np.cross(p2,q2)
    l10 = np.cross(p2,s2)
    l9 = l1/l1[2]
    l10 = l2/l2[2]

    l11 = np.cross(p2,s2)
    l12 = np.cross(s2,r2)
    l11 = l11/l11[2]
    l12 = l12/l12[2]

    l13 = np.cross(s2,r2)
    l14 = np.cross(r2,q2)
    l13 = l13/l13[2]
    l14 = l14/l14[2]



    lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14]


    A = np.zeros((7,5))
    b = np.zeros((7,1))
    for i in range(7):
        l = lines[2*i]
        m = lines[2*i+1]
        A[i][0:5] = [l[0]*m[0], (l[0]*m[1]+l[1]*m[0])/2, l[1]*m[1], (l[0]*m[2]+l[2]*m[0])/2, (l[1]*m[2]+l[2]*m[1])/2]
        b[i] = -l[2]*m[2]
    s = np.matmul(np.linalg.pinv(A),b)
    s = s/np.max(s)
    S = np.zeros((2,2))
    S[0,0] = s[0]
    S[0,1] = s[1]/2
    S[1,0] = s[1]/2
    S[1,1] = s[2]
    V,D,V_h = np.linalg.svd(S)
    D = np.sqrt(np.diag(D))
    A_mat = np.matmul(np.matmul(V,D),V.transpose())
    bb = np.array([s[3]/2, s[4]/2])
    v = np.matmul(np.linalg.pinv(A_mat),bb)

    H = np.zeros((3,3))
    H[0,0] = A_mat[0,0]
    H[0,1] = A_mat[0,1]
    H[1,0] = A_mat[1,0]
    H[1,1] = A_mat[1,1]
    H[2,0] = v[0]
    H[2,1] = v[1]
    H[2,2] = 1

    return H

def rectImage(range_img, H):
    # Inverse of H
    H_inv = np.linalg.pinv(H)
    H_inv = H_inv/H_inv[2,2]
    # Get world coordinates 
    world_coords = []
    img_coords = [(0, 0), (0, range_img.shape[0]-1), (range_img.shape[1]-1, range_img.shape[0]-1), (range_img.shape[1]-1, 0)]   
    for i in img_coords:
        img_coord = np.array((i[0], i[1], 1))
        world_coord = np.matmul(H, img_coord)
        world_coord = world_coord/world_coord[2]
        world_coords.append([world_coord[0],world_coord[1]])
    
    # Get size of world image
    x_max = int(max([c[0] for c in world_coords]))
    x_min = int(min([c[0] for c in world_coords]))
    y_max = int(max([c[1] for c in world_coords]))
    y_min = int(min([c[1] for c in world_coords]))
    x_length = x_max - x_min
    y_length = y_max - y_min
    world_img = np.zeros((y_length, x_length, 3))
    for row in range(0, y_length):
        for col in range(0, x_length):
            domain_coord = np.array((col+x_min, row+y_min, 1))
            range_coord = np.matmul(H_inv,domain_coord) 
            range_coord = range_coord / range_coord[2]
            range_coord = range_coord.astype(int)
            if (range_coord[0]<(range_img.shape[1]-1) and range_coord[1]<(range_img.shape[0]-1) and range_coord[0]>0 and range_coord[1]>0 ):
                world_img[row, col] = range_img[range_coord[1], range_coord[0]]
    return world_img

def getRectImage(img, coords1, coords2):
    boxImg = drawCircleNBox(img, coords1, coords2)
    
    # Get the Homography
    H = getH(coords1,coords2)
    H_inv = np.linalg.pinv(H)
    H_inv = H_inv/H_inv[2,2]
    
    # Get the rectified image
    img_Rect  = rectImage(img, H_inv)

    return boxImg, img_Rect

#----------------------------------Task---------------------------------#
# Input Images
building_img = cv2.imread('HW3/building.jpg')

# Coordinates
building_pqrs = [(346, 227),(345,378),(460,387),(460,256)]
building_pqrs2 = [(240,200),(236,367),(295,373),(298,216)]

building_boxImg, building_img_Rect = getRectImage(building_img,building_pqrs,building_pqrs2)

cv2.imwrite('HW3/building_box_img2.jpeg',building_boxImg) 
cv2.imwrite('HW3/building_Rect1.3.jpeg',building_img_Rect)

# Input Image
nighthawks_img = cv2.imread('HW3/nighthawks.jpg')

# Coordinates
nighthawks_pqrs = [(76, 180), (78, 653), (805, 620), (802, 219)]
nighthawks_pqrs2 = [(12,100),(14,729),(865,678),(863,162)]

nighthawks_boxImg,nighthawks_img_Rect = getRectImage(nighthawks_img, nighthawks_pqrs,nighthawks_pqrs2)

cv2.imwrite('HW3/nighthawks_box_img2.jpeg', nighthawks_boxImg) 
cv2.imwrite('HW3/nighthawks_Rect1.3.jpeg', nighthawks_img_Rect)

# Input Image
paintings_img = cv2.imread('HW3/paintings.jpg')

# Coordinates
paintings_pqrs = [(537,465),(538,947),(940,842),(933,485)]
paintings_pqrs2 = [(626,110),(628,275),(849,314),(847,173)]

paintings_boxImg, paintings_img_Rect = getRectImage(paintings_img,paintings_pqrs,paintings_pqrs2)

cv2.imwrite('HW3/paintings_boxImg2.jpeg', paintings_boxImg) 
cv2.imwrite('HW3/paintings_img_Rect1.3.jpeg', paintings_img_Rect)

# Input Image
calendar_img = cv2.imread('HW3/calendar.jpg')

# Coordinates
calendar_pqrs = [(459,222),(491,973),(672,933),(652,241)]
calendar_pqrs2 = [(412,266),(420,451),(819,454),(816,298)]

calendar_boxImg, calendar_img_Rect = getRectImage(calendar_img,calendar_pqrs,calendar_pqrs2)

cv2.imwrite('HW3/calendar_boxImg2.jpeg', calendar_boxImg) 
cv2.imwrite('HW3/calendar_img_Rect1.3.jpeg', calendar_img_Rect)



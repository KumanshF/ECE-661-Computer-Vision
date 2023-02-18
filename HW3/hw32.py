import numpy as np
import cv2

#-------------------------Functions-------------------------------#
def drawCircleNBox(img, coords):
    img_box = img.copy()
    #draw circles at the coords
    for x,y in coords:
        cv2.circle(img_box, (x,y), 8, (255, 0, 0), -1)

    #draw lines connecting the coords to form a box
    coords.append(coords[0])
    for i in range(4):
        if i == 0 or i==2 or i==4:
            cv2.line(img_box, coords[i], coords[i+1], (0, 255, 0), 4)
        else:
            cv2.line(img_box, coords[i], coords[i+1], (0, 0, 255), 4)
    
    #return Img
    return img_box

def getProjectiveH(coords):
    p1 = (coords[0][0],coords[0][1],1)
    p2 = (coords[1][0],coords[1][1],1)
    p3 = (coords[2][0],coords[2][1],1)
    p4 = (coords[3][0],coords[3][1],1)

    l1 = np.cross(p1,p2)
    l2 = np.cross(p3,p4)
    vp_1 = np.cross(l1,l2)
    vp_1 = vp_1/vp_1[2]

    l3 = np.cross(p1,p4)
    l4 = np.cross(p2,p3)
    vp_2 = np.cross(l3,l4)
    vp_2 = vp_2/vp_2[2]

    vl = np.cross(vp_2, vp_1)
    vl = vl/vl[2]
    
    H = np.zeros((3,3))
    H[0,0] = 1
    H[1,1] = 1
    H[2] = vl

    return H

def getAffineH(coords):
    p1 = (coords[0][0],coords[0][1],1)
    p2 = (coords[1][0],coords[1][1],1)
    p3 = (coords[2][0],coords[2][1],1)
    p4 = (coords[3][0],coords[3][1],1)
    
    # Get parallel lines
    l1 = np.cross(p1,p2)
    m1 = np.cross(p1,p4)
    l1 = l1/l1[2]
    m1 = m1/m1[2]

    l2 = np.cross(p2,p3)
    m2 = np.cross(p3,p4)
    l2 = l2/l2[2]
    m2 = m2/m2[2]

    A = np.zeros((2,2))
    b = np.zeros((2,1))
    A = [[l1[0]*m1[0], l1[0]*m1[1]+l1[1]*m1[0]],[l2[0]*m2[0], l2[0]*m2[1]+l2[1]*m2[0]]]
    b = [[-l1[1]*m1[1]],[-l2[1]*m2[1]]]

    s = np.matmul(np.linalg.pinv(A),b)
    S = np.zeros((2,2))
    S[0,0] = s[0]
    S[0,1] = s[1]
    S[1,0] = s[1]
    S[1,1] = 1
    V,D,V_h = np.linalg.svd(S)
    D = np.sqrt(np.diag(D))
    A_mat = np.matmul(np.matmul(V,D),V.transpose())
    H = np.zeros((3,3))
    H[0,0] = A_mat[0,0]
    H[0,1] = A_mat[0,1]
    H[1,0] = A_mat[1,0]
    H[1,1] = A_mat[1,1]
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
    print(y_length, x_length)
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

def getRectImg(dom_img_coords, range_img):
    # Draw circles and bounding box on card
    boxImg = drawCircleNBox(range_img, dom_img_coords)

    # Removing Projective Distortion
    H_proj = getProjectiveH(dom_img_coords)
    project_img_rect = rectImage(range_img, H_proj)

    dom_coords_affine = []
    for c in dom_img_coords:
        coord = np.array((c[0], c[1], 1))
        coords = np.matmul(H_proj, coord)
        coords = coords/coords[2]
        coords = coords.astype(int)
        dom_coords_affine.append([coords[0], coords[1]])
    print("newcoords:", dom_coords_affine)
    
    # Removing Affine Distortion
    H_affine = getAffineH(dom_coords_affine)

    H_projAff = np.matmul(np.linalg.pinv(H_affine),H_proj)
    # Removing both distortion
    img_rect = rectImage(range_img, H_projAff)

    return boxImg, project_img_rect, img_rect

#----------------------------------Task---------------------------------#
# Input Images
building_img = cv2.imread('HW3/building.jpg')
nighthawks_img = cv2.imread('HW3/nighthawks.jpg')

building_pqrs = [(242, 130), (657, 273), (660, 403), (236, 369)]
nighthawks_pqrs = [(76, 180), (78, 653), (805, 620), (802, 219)]

paintings_img = cv2.imread('HW3/paintings.jpg')
calendar_img = cv2.imread('HW3/calendar.jpg')

paintings_pqrs = [(537,465),(538,947),(940,842),(933,485)]
calendar_pqrs = [(459,222),(491,973),(672,933),(652,241)]

building_boxImg, building_img_ProjRect, building_img_Rect  = getRectImg(building_pqrs, building_img)
nighthawks_boxImg, nighthawks_img_ProjRect, nighthawks_img_Rect  = getRectImg(nighthawks_pqrs, nighthawks_img)
paintings_boxImg, paintings_img_ProjRect, paintings_img_Rect  = getRectImg(paintings_pqrs, paintings_img)
calendar_boxImg, calendar_img_ProjRect, calendar_img_Rect  = getRectImg(calendar_pqrs, calendar_img)


cv2.imwrite('HW3/building_box_img1.jpeg',building_boxImg) 
cv2.imwrite('HW3/building_ProjectRect.jpeg',building_img_ProjRect)
cv2.imwrite('HW3/building_Rect1.2.jpeg',building_img_Rect)
cv2.imwrite('HW3/nighthawks_boxImg1.jpeg',nighthawks_boxImg) 
cv2.imwrite('HW3/nighthawks_ProjRect.jpeg',nighthawks_img_ProjRect)
cv2.imwrite('HW3/nighthawks_Rect1.2.jpeg',nighthawks_img_Rect)

cv2.imwrite('HW3/paintings_boxImg1.jpeg',paintings_boxImg) 
cv2.imwrite('HW3/paintings_img_ProjRect.jpeg',paintings_img_ProjRect)
cv2.imwrite('HW3/paintings_img_Rect1.2.jpeg',paintings_img_Rect)
cv2.imwrite('HW3/calendar_boxImg1.jpeg',calendar_boxImg) 
cv2.imwrite('HW3/calendar_img_ProjRect.jpeg',calendar_img_ProjRect)
cv2.imwrite('HW3/calendar_img_Rect1.2.jpeg',calendar_img_Rect)

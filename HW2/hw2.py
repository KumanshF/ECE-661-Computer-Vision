import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image

#-------------------------Functions-------------------------------#
def drawCircleNBox(img, coords):
    img_box = img.copy()
    #draw circles at the coords
    for x,y in coords:
        cv2.circle(img_box, (x,y), 8, (255, 0, 0), -1)

    #draw lines connecting the coords to form a box
    coords.append(coords[0])    
    cv2.polylines(img_box, [np.array(coords)], True, (255, 0, 0), 2)
    
    #return Img
    return img_box

def getMask(img, coords):
    img_mask = img.copy()
    img_mask = cv2.fillPoly(img_mask, [np.array(coords)], (0, 0, 0))
    return img_mask

def calcHomography(dom_coords, range_coords):
    ## AH = b

    # A and B Matrix
    A = np.zeros((8,8))
    b = np.zeros((8,1))
    for i in range(4):
        xp = range_coords[i][0]
        yp = range_coords[i][1]
        x = dom_coords[i][0]
        y = dom_coords[i][1]
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

def calcAffineHomography(dom_coords, range_coords):
    ## AH = b

    # A and B Matrix
    A = np.zeros((8,8))
    b = np.zeros((8,1))
    for i in range(4):
        xp = range_coords[i][0]
        yp = range_coords[i][1]
        x = dom_coords[i][0]
        y = dom_coords[i][1]
        # A matrix
        A[2*i][0:8] = [x, y, 1 , 0 , 0 , 0 ,0, 0]
        A[2*i+1][0:8] = [0, 0, 0, x, y, 1, 0, 0]  
        # B matrix
        b[2*i] = xp
        b[2*i+1] = yp
    
    # H matrix
    H = np.zeros((3,3))
    H = np.matmul(np.linalg.pinv(A),b)
    H = np.append(H,[1])
    H = np.reshape(H, (3,3))
    return H

    
def mappingHom(img_x, img_xp, mask_img, H):
    img_mapped = img_xp.copy()
    for row in range(img_xp.shape[1]):
        for col in range(img_xp.shape[0]):
            if any(mask_img[col][row])==0:
                domain_coord = np.array((row, col, 1))
                range_coord = np.matmul(H,domain_coord) 
                range_coord = range_coord / range_coord[2]
                range_coord = range_coord.astype(int)
                if (range_coord[0]<img_x.shape[1] and range_coord[1]<img_x.shape[0]):
                    img_mapped[col, row] = img_x[range_coord[1], range_coord[0]]
    return img_mapped        

def getMappedImg(dom_img, dom_img_coords, range_img, range_img_coords):
    # draw circles and bounding box on card
    boxImg = drawCircleNBox(range_img, range_img_coords)

    # Create a mask image
    range_img_mask = getMask(range_img, range_img_coords) 

    # Generate a Homography Matrix
    H = calcHomography(dom_img_coords, range_img_coords)
    #Inverse the H to go from range to domain
    H = np.linalg.pinv(H)

    # Map points from domain to range
    mappedImg = mappingHom(dom_img, range_img, range_img_mask, H)

    return boxImg, range_img_mask, mappedImg

def getMappedImg_affine(dom_img, dom_img_coords, range_img, range_img_coords):
    # draw circles and bounding box on card
    boxImg = drawCircleNBox(range_img, range_img_coords)

    # Create a mask image
    range_img_mask = getMask(range_img, range_img_coords) 

    # Generate a Homography Matrix
    H = calcAffineHomography(dom_img_coords, range_img_coords)
    #Inverse the H to go from range to domain
    H = np.linalg.pinv(H)

    # Map points from domain to range
    mappedImg = mappingHom(dom_img, range_img, range_img_mask, H)

    return boxImg, range_img_mask, mappedImg

# -----------------------------Task 1.1------------------------------#

# Image Coordinates
car_coords = [(0,0), (0, 558), (760, 558), (760,0)]
card1_coords = [(488, 252), (612, 1112),(1222, 798), (1242, 177)]
card2_coords = [(319, 228), (210, 860), (874, 1128), (1042, 229)]
card3_coords = [(584, 48), (62, 590), (702, 1213), (1228, 674)]

# Input Images
car = cv2.imread('car.jpg')
img_card1 = cv2.imread('card1.jpeg')
img_card2 = cv2.imread('card2.jpeg')
img_card3 = cv2.imread('card3.jpeg')

#----------Projecting car on card 1-----------------#
card1_boxImg, card1_mask, card1_mapped = getMappedImg(car, car_coords, img_card1, card1_coords)

#----------Projecting car on card 2-----------------#
card2_boxImg, card2_mask, card2_mapped = getMappedImg(car, car_coords, img_card2, card2_coords)

#----------Projecting car on card 3-----------------#
card3_boxImg, card3_mask, card3_mapped = getMappedImg(car, car_coords, img_card3, card3_coords)


# Save images
cv2.imwrite('card1_boxed.jpeg',card1_boxImg)
cv2.imwrite('card1_mask.jpeg', card1_mask)
cv2.imwrite('card1_mapped.jpeg',card1_mapped)

cv2.imwrite('card2_boxed.jpeg',card2_boxImg)
cv2.imwrite('card2_mask.jpeg', card2_mask)
cv2.imwrite('card2_mapped.jpeg',card2_mapped)

cv2.imwrite('card3_boxed.jpeg',card3_boxImg)
cv2.imwrite('card3_mask.jpeg', card3_mask)
cv2.imwrite('card3_mapped.jpeg',card3_mapped)



# -----------------------------Task 1.2------------------------------#
# Calculating H_ab
H_ab = calcHomography(card1_coords, card2_coords)

# Calculating H_bc
H_bc = calcHomography(card2_coords, card3_coords)

# Calculating H_ac
H_ac = np.matmul(H_ab, H_bc)

# Calculating H_ac inv
H_ac_inv = np.linalg.inv(H_ac)

# Mapping 1a to 1c
mappedImg3 = np.zeros(img_card3.shape, dtype='uint8')
for row in range(img_card3.shape[1]):
    for col in range(img_card3.shape[0]):
        domain_coord = np.array((row, col, 1))
        range_coord = np.matmul(H_ac_inv,domain_coord) 
        range_coord = range_coord / range_coord[2]
        range_coord = range_coord.astype(int)
        if (range_coord[0]<img_card1.shape[1] and range_coord[1]<img_card1.shape[0] and range_coord[0]>0 and range_coord[1]>0):
            mappedImg3[col, row] = img_card1[range_coord[1], range_coord[0]]

# Save image
cv2.imwrite('card1_3.jpeg',mappedImg3)

#-----------------------------Task 1.3------------------------------#
# Map the images
card1_boxImg_affine, card1_mask_affine, card1_mapped_affine = getMappedImg_affine(car, car_coords, img_card1, card1_coords)
card2_boxImg_affine, card2_mask_affine, card2_mapped_affine = getMappedImg_affine(car, car_coords, img_card2, card2_coords)
card3_boxImg_affine, card3_mask_affine, card3_mapped_affine = getMappedImg_affine(car, car_coords, img_card3, card3_coords)

# Save images
cv2.imwrite('card1_boxed_affine.jpeg',card1_boxImg_affine)
cv2.imwrite('card1_mask_affine.jpeg', card1_mask_affine)
cv2.imwrite('card1_mapped_affine.jpeg',card1_mapped_affine)

cv2.imwrite('card2_boxed_affine.jpeg',card2_boxImg_affine)
cv2.imwrite('card2_mask_affine.jpeg', card2_mask_affine)
cv2.imwrite('card2_mapped_affine.jpeg',card2_mapped_affine)

cv2.imwrite('card3_boxed_affine.jpeg',card3_boxImg_affine)
cv2.imwrite('card3_mask_affine.jpeg', card3_mask_affine)
cv2.imwrite('card3_mapped_affine.jpeg',card3_mapped_affine)

#--------------------------------------Task 2.1------------------------------------------#

# Image Coordinates
amongus_coords = [(0,0), (0,511), (511,511), (511, 0)]
frame1_coords = [(991,850), (1047,2553), (2253, 2347), (2261, 1017)]
frame2_coords = [(950,1245), (968, 2520), (2045, 2774), (2045, 1003)]
frame3_coords = [(590, 943), (640, 2691), (2349, 2681), (2370, 932)]

# Input Images
amongus_img = cv2.imread('amongus.jpg')
frame1_img = cv2.imread('frame1.jpg')
frame2_img = cv2.imread('frame2.jpg')
frame3_img = cv2.imread('frame3.jpg')

#----------Projecting amongUs on frame 1-----------------#
frame1_boxImg, frame1_mask, frame1_mapped = getMappedImg(amongus_img, amongus_coords, frame1_img, frame1_coords)

#----------Projecting amongUs on frame 2-----------------#
frame2_boxImg, frame2_mask, frame2_mapped = getMappedImg(amongus_img, amongus_coords, frame2_img, frame2_coords)

#----------Projecting amongUs on frame 3-----------------#
frame3_boxImg, frame3_mask, frame3_mapped = getMappedImg(amongus_img, amongus_coords, frame3_img, frame3_coords)


# # Save images
cv2.imwrite('frame1_boxImg.jpeg',frame1_boxImg)
cv2.imwrite('frame1_mask.jpeg', frame1_mask)
cv2.imwrite('frame1_mapped.jpeg',frame1_mapped)

cv2.imwrite('frame2_boxImg.jpeg',frame2_boxImg)
cv2.imwrite('frame2_mask.jpeg', frame2_mask)
cv2.imwrite('frame2_mapped.jpeg',frame2_mapped)

cv2.imwrite('frame3_boxImg.jpeg',frame3_boxImg)
cv2.imwrite('frame3_mask.jpeg', frame3_mask)
cv2.imwrite('frame3_mapped.jpeg',frame3_mapped)

# -----------------------------Task 2.2------------------------------#
# Calculating H_ab
H_ab = calcHomography(frame1_coords, frame2_coords)

# Calculating H_bc
H_bc = calcHomography(frame2_coords, frame3_coords)

# Calculating H_ac
H_ac = np.matmul(H_ab, H_bc)

# Calculating H_ac inv
H_ac_inv = np.linalg.inv(H_ac)

# Mapping 7a to 7c
mappedFrame = np.zeros(frame3_img.shape, dtype='uint8')
for row in range(frame3_img.shape[1]):
    for col in range(frame3_img.shape[0]):
        domain_coord = np.array((row, col, 1))
        range_coord = np.matmul(H_ac_inv,domain_coord) 
        range_coord = range_coord / range_coord[2]
        range_coord = range_coord.astype(int)
        if (range_coord[0]<frame1_img.shape[1] and range_coord[1]<frame1_img.shape[0] and range_coord[0]>0 and range_coord[1]>0):
            mappedFrame[col, row] = frame1_img[range_coord[1], range_coord[0]]

# Save image
cv2.imwrite('frame1_3.jpeg',mappedFrame)

#-----------------------------Task 2.3------------------------------#
# Map the images
frame1_boxImg_affine, frame1_mask_affine, frame1_mapped_affine = getMappedImg_affine(amongus_img, amongus_coords, frame1_img, frame1_coords)
frame2_boxImg_affine, frame2_mask_affine, frame2_mapped_affine = getMappedImg_affine(amongus_img, amongus_coords, frame2_img, frame2_coords)
frame3_boxImg_affine, frame3_mask_affine, frame3_mapped_affine = getMappedImg_affine(amongus_img, amongus_coords, frame3_img, frame3_coords)

cv2.imwrite('frame1_boxed_affine.jpeg',frame1_boxImg_affine)
cv2.imwrite('frame1_mask_affine.jpeg', frame1_mask_affine)
cv2.imwrite('frame1_mapped_affine.jpeg',frame1_mapped_affine)

cv2.imwrite('frame2_boxed_affine.jpeg',frame2_boxImg_affine)
cv2.imwrite('frame2_mask_affine.jpeg', frame2_mask_affine)
cv2.imwrite('frame2_mapped_affine.jpeg',frame2_mapped_affine)

cv2.imwrite('frame3_boxed_affine.jpeg',frame3_boxImg_affine)
cv2.imwrite('frame3_mask_affine.jpeg', frame3_mask_affine)
cv2.imwrite('frame3_mapped_affine.jpeg',frame3_mapped_affine)
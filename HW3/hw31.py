import numpy as np
import cv2

#-------------------------Functions-------------------------------#
def drawCircleNBox(img, coords):
    img_box = img.copy()
    #draw circles at the coords
    for x,y in coords:
        cv2.circle(img_box, (x,y), 10, (255, 0, 0), -1)

    #draw lines connecting the coords to form a box
    coords.append(coords[0])    
    cv2.polylines(img_box, [np.array(coords)], True, (255, 0, 0), 5)
    
    #return Img
    return img_box

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

def getRectImg(dom_img_coords, range_img_coords, range_img):
    # Draw circles and bounding box on card
    boxImg = drawCircleNBox(range_img, dom_img_coords)

    # Generate a Homography Matrix
    H = calcHomography(dom_img_coords, range_img_coords)

    # Map range space (image) points to domain space (real world) points using H_inv
    img_rect = rectImage(range_img, H)

    return boxImg, img_rect

def rectImage(range_img, H):
    # Inverse of H
    H_inv = np.linalg.pinv(H)

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
            #print(domain_coord,range_coord)
            if (range_coord[0]<(range_img.shape[1]-1) and range_coord[1]<(range_img.shape[0]-1) and range_coord[0]>0 and range_coord[1]>0 ):
                world_img[row, col] = range_img[range_coord[1], range_coord[0]]
    return world_img
# -----------------------------Task 1------------------------------#

# Image Coordinates
building_img_coords = [(240,200), (236, 368), (294, 374), (298,214)]
building_irl_coords = [(0, 0), (0, 450),(150, 450), (150, 0)]
nighthawks_img_coords = [(76, 180), (78, 653), (805, 620), (802, 219)]
nighthawks_irl_coords = [(0, 0), (0, 85), (150, 85), (150, 0)]

paintings_img_coords = [(537,465),(538,947),(940,842),(933,485)]
paintings_irl_coords = [(0,0),(0,350),(400,350),(400,0)]
calendar_img_coords = [(459,222),(491,973),(672,933),(652,241)]
calendar_irl_coords = [(0,0),(0,340),(100,340),(100,0)]

# Input Images
building_img = cv2.imread('HW3/building.jpg')
nighthawks_img = cv2.imread('HW3/nighthawks.jpg')

paintings_img = cv2.imread('HW3/paintings.jpg')
calendar_img = cv2.imread('HW3/calendar.jpg')

#----------------------------Task1.1: p-t-p correspondence------------------------------#
building_boxImg, building_imgRect = getRectImg(building_img_coords, building_irl_coords, building_img)
nighthawks_boxImg, nighthawks_imgRect = getRectImg(nighthawks_img_coords, nighthawks_irl_coords, nighthawks_img)

paintings_boxImg, paintings_imgRect = getRectImg(paintings_img_coords, paintings_irl_coords, paintings_img)
calendar_boxImg, calendar_imgRect = getRectImg(calendar_img_coords, calendar_irl_coords, calendar_img)


# Show and save Images
cv2.imwrite('HW3/building_boxImg.jpeg',building_boxImg)
cv2.imwrite('HW3/building_imgRect.jpeg',building_imgRect)
cv2.imwrite('HW3/nighthawks_boxImg.jpeg',nighthawks_boxImg)
cv2.imwrite('HW3/nighthawks_imgRect.jpeg',nighthawks_imgRect)
cv2.imwrite('HW3/paintings_boxImg.jpeg',paintings_boxImg)
cv2.imwrite('HW3/paintings_imgRect.jpeg',paintings_imgRect)
cv2.imwrite('HW3/calendar_boxImg.jpeg',calendar_boxImg)
cv2.imwrite('HW3/calendar_imgRect.jpeg',calendar_imgRect)


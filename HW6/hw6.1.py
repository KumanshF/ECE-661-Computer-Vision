import numpy as np
import cv2
import math

#---------------------------Functions-----------------------------#
def RGBImageSegmentation(image, iters, flip, imgName):
    img = image.copy()
    
    # Get RGB layers of the img
    channels = cv2.split(img) # B - G - R
    # Create Masks array to store mask for each of the 3 layers
    masks = []
    mask_all = np.ones(channels[0].shape).astype(np.uint8)
    for c, channel_data in enumerate(channels):
        # Update mask
        mask = getMask(channel_data, iters[c], flip[c])
        masks.append(mask)
        mask_all = np.logical_and(mask_all, mask)
    # Foreground with blue, gree, red mask
    mask_blue = np.uint8(masks[0]*255)
    mask_green = np.uint8(masks[1]*255)
    mask_red = np.uint8(masks[2]*255)
    # Foreground with merged masks
    mask_comb_img = np.uint8(mask_all*255)

    # Save Images
    cv2.imwrite('HW6/'+imgName+'_blue_img.jpeg', channels[0])
    cv2.imwrite('HW6/'+imgName+'_green_img.jpeg', channels[1])
    cv2.imwrite('HW6/'+imgName+'_red_img.jpeg', channels[2])

    cv2.imwrite('HW6/'+imgName+'_blue_mask.jpeg', mask_blue)
    cv2.imwrite('HW6/'+imgName+'_green_mask.jpeg', mask_green)
    cv2.imwrite('HW6/'+imgName+'_red_mask.jpeg', mask_red)
    cv2.imwrite('HW6/'+imgName+'_mask_all.jpeg', mask_comb_img)

    return mask_blue, mask_green, mask_red, mask_all


def getMask(img_data, iters, flip):
    data = img_data.flatten()

    for i in range(iters):
        # Get the threshold using Otsu's Algorithm
        threshold = OtsuThresh(data)

        # Create a mask to update
        mask = np.zeros(img_data.shape, dtype = np.uint8)
        
        # Using the threshold, update the mask
        if flip:
            mask[img_data > threshold] = 1
            data = [i for i in data if i>threshold]
        else:
            mask[img_data <= threshold] = 1
            data = [i for i in data if i<threshold]

        data = np.asarray(data)    
    return mask

def OtsuThresh(img):
    # get histogram
    hist, bins = np.histogram(img, bins = 256, range = (0,256))
    sum_t = sum(hist*bins[:-1])
    back = 0
    sumback = 0    
    best_fn = -1
    threshold = 0
    for i in range(256):
        back, fore = sum(hist[:i]), sum(hist[i+1:])
        if back == 0 or fore == 0:
            continue
        sumback += i*hist[i]
        sumfore = np.int32(sum_t - sumback)
        fn = back*fore*(sumback/back - sumfore/fore)**2
        if fn>=best_fn:
            threshold = i
            best_fn = fn

    print(threshold)
    return threshold

def textureSegmentation(image, window, iters, flip, imgName):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    layers = []
    masks = []
    mask_all = np.ones(img.shape).astype(np.uint8)
    for i, N in enumerate(window):
        layer = getTexturelayer(img, N)
        layers.append(layer)
        mask = getMask(layer, iters[i], flip[i])
        masks.append(mask)
        mask_all = np.logical_and(mask_all, mask)
    # Foreground with blue, gree, red mask
    mask_1 = np.uint8(masks[0]*255)
    mask_2 = np.uint8(masks[1]*255)
    mask_3 = np.uint8(masks[2]*255)
    # Foreground with merged masks
    mask_comb_img = np.uint8(mask_all*255)

    # Save Images
    cv2.imwrite('HW6/'+imgName+'_textLayer1.jpeg', layers[0])
    cv2.imwrite('HW6/'+imgName+'_textLayer2.jpeg', layers[1])
    cv2.imwrite('HW6/'+imgName+'_textLayer3.jpeg', layers[2])
    cv2.imwrite('HW6/'+imgName+'_layer1Mask.jpeg', mask_1)
    cv2.imwrite('HW6/'+imgName+'_layer2Mask.jpeg', mask_2)
    cv2.imwrite('HW6/'+imgName+'_layer3Mask.jpeg', mask_3)
    cv2.imwrite('HW6/'+imgName+'_textureMask_all.jpeg', mask_comb_img)

    return mask_1, mask_2, mask_3, mask_all

def getTexturelayer(img, N):
    varLayer = np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mid = int((N-1)/2)
            u = max(0, y-mid)
            d = min(img.shape[0], y + mid + 1)
            l = max(0, x-mid)
            r = min(img.shape[1], x + mid + 1)
            window = img[u:d, l:r]
            varLayer[y][x] = np.var(window)
    varLayer = np.uint8(np.round(255 * varLayer / (np.max(varLayer)-np.min(varLayer)) ))
    return varLayer

def opening(imgName, mask, erosion, dilation, eros_size, eros_iter, dil_size, dil_iter):
     # Use dilation and Erosion to clean the mask
    if erosion:
        k = np.ones((eros_size,eros_size), np.uint8)
        mask = cv2.erode(np.float32(mask), k, iterations=eros_iter)
        # cv2.imshow('frame1', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('HW6/'+imgName+'_opening_eros_img.jpeg', mask*255)
    if dilation:
        k = np.ones((dil_size, dil_size), np.uint8)
        mask = cv2.dilate(np.float32(mask), k, iterations=dil_iter)
        # cv2.imshow('frame2', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('HW6/'+imgName+'_opening_dil_eros_img.jpeg', mask*255)
    return mask

def closing(imgName, mask, erosion, dilation, eros_size, eros_iter, dil_size, dil_iter):
     # Use dilation and Erosion to clean the mask
    if dilation:
        k = np.ones((dil_size, dil_size), np.uint8)
        mask = cv2.dilate(np.float32(mask), k, iterations=dil_iter)
        # cv2.imshow('frame2', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('HW6/'+imgName+'_closing_dilation_img.jpeg', mask*255)
    if erosion:
        k = np.ones((eros_size,eros_size), np.uint8)
        mask = cv2.erode(np.float32(mask), k, iterations=eros_iter)
        # cv2.imshow('frame1', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('HW6/'+imgName+'_closing_eros_dil_img.jpeg', mask*255)
    return mask

def getContour(imgName, mask):
    contour = np.zeros(mask.shape, dtype = np.uint8)
    for y in range(1,mask.shape[0]-1):
        for x in range(1,mask.shape[1]-1):
            if mask[y][x] == 0:
                continue
            window = mask[y-1:y+2, x-1:x+2]
            if sum(window.flatten())<9:
                contour[y][x] = 1
    contour_img = np.uint8(contour*255)

    # Save Image
    cv2.imwrite('HW6/'+imgName+'_contour_img.jpeg', contour_img)

    return contour_img


#-----------------------------Task 1------------------------------#

# Input Images
cat1 = cv2.imread('HW6/cat.jpg')
car1 = cv2.imread('HW6/car.jpg')
cat2 = cv2.imread('HW6/cat2.jpg')
car2 = cv2.imread('HW6/car2.jpg')


#----------------Task 1-------------------#

#----------Cat1 Image-----------#

# BGR Segmentation and Contouring
blue_mask, green_mask, red_mask, mask_all_bgrSeg = RGBImageSegmentation(cat1, iters = [1,1,1], flip = [0,0,1], imgName="cat1")
mask_all_opening = closing("cat1_bgrSeg", mask_all_bgrSeg, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_closing = opening("cat1_bgrSeg", mask_all_bgrSeg, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("cat1_bgrSeg", mask_all_bgrSeg)
mask_all = getContour("cat1_bgrSeg_opening", mask_all_opening)
mask_all = getContour("cat1_bgrSeg_closing", mask_all_closing)

# Texture Segmentation and Contouring
layer3Mask, layer5Mask, layer7Mask, textureMask_all = textureSegmentation(cat1, window = [3,5,7], iters = [1,1,1], flip = [0,1,1], imgName="cat1")
mask_all_closing = closing("cat1_texture", textureMask_all, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_opening = opening("cat1_texture", textureMask_all, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("cat1_texture", textureMask_all)
mask_all = getContour("cat1_texture_opening", mask_all_opening)
mask_all = getContour("cat1_texture_closing", mask_all_closing)


#----------Car1 Image-----------#

# BGR Segmentation and Contouring
blue_mask, green_mask, red_mask, mask_all_seg = RGBImageSegmentation(car1, iters = [1,1,1], flip=[0,1,0], imgName="car1")
mask_all_closing = closing("car1_bgrSeg", mask_all_seg, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_opening = opening("car1_bgrSeg", mask_all_seg, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("car1_bgrSeg", mask_all_seg)
mask_all = getContour("car1_bgrSeg_opening", mask_all_opening)
mask_all = getContour("car1_bgrSeg_closing", mask_all_closing)

layer1Mask, layer2Mask, layer3Mask, textureMask_all = textureSegmentation(car1, window = [3,5,7], iters = [1,1,1], flip = [0,0,1], imgName="car1")
mask_all_closing = closing("car1_texture", textureMask_all, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_opening = opening("car1_texture", textureMask_all, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("car1_texture", textureMask_all)
mask_all = getContour("car1_texture_closing", mask_all_closing)
mask_all = getContour("car1_texture_opening", mask_all_opening)


#----------------Task 2-------------------#

#----------Cat2 Image-----------#

# BGR Segmentation and Contouring
blue_mask, green_mask, red_mask, mask_all_bgrSeg = RGBImageSegmentation(cat2, iters = [1,1,2], flip = [1,1,1], imgName="cat2")
mask_all_opening = closing("cat2_bgrSeg", mask_all_bgrSeg, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_closing = opening("cat2_bgrSeg", mask_all_bgrSeg, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("cat2_bgrSeg", mask_all_bgrSeg)
mask_all = getContour("cat2_bgrSeg_opening", mask_all_opening)
mask_all = getContour("cat2_bgrSeg_closing", mask_all_closing)

# Texture Segmentation and Contouring
layer3Mask, layer5Mask, layer7Mask, textureMask_all = textureSegmentation(cat2, window = [5,7,9], iters = [1,1,1], flip = [1,1,1], imgName="cat2")
mask_all_closing = closing("cat2_texture", textureMask_all, erosion=1, dilation=1, eros_size=2, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_opening = opening("cat2_texture", textureMask_all, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("cat2_texture", textureMask_all)
mask_all = getContour("cat2_texture_opening", mask_all_opening)
mask_all = getContour("cat2_texture_closing", mask_all_closing)


#----------Car2 Image-----------#
# BGR Segmentation and Contouring
blue_mask, green_mask, red_mask, mask_all_seg = RGBImageSegmentation(car2, iters = [2,2,2], flip=[0,0,0], imgName="car2")
mask_all_closing = closing("car2_bgrSeg", mask_all_seg, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_opening = opening("car2_bgrSeg", mask_all_seg, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("car2_bgrSeg", mask_all_seg)
mask_all = getContour("car2_bgrSeg_opening", mask_all_opening)
mask_all = getContour("car2_bgrSeg_closing", mask_all_closing)

# Texture Segmentation and Contouring
layer1Mask, layer2Mask, layer3Mask, textureMask_all = textureSegmentation(car2, window = [5,7,9], iters = [1,1,1], flip = [1,1,1], imgName="car2")
mask_all_closing = closing("car2_texture", textureMask_all, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all_opening = opening("car2_texture", textureMask_all, erosion=1, dilation=1, eros_size=1, eros_iter=1, dil_size=3, dil_iter=2)
mask_all = getContour("car2_texture", textureMask_all)
mask_all = getContour("car2_texture_closing", mask_all_closing)
mask_all = getContour("car2_texture_opening", mask_all_opening)

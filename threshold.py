#!/usr/bin/python3
import cv2
import glob
import numpy as np


# Function to get gradient of an image along x or y direction
def gradientSobel(img, orient='x', k_size=3, thresh=(0,255)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if(orient == 'x'):
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
    elif(orient == 'y'):
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
    sobel_abs = np.absolute(sobel)
    sobel_scale = np.uint8((sobel_abs * 255)/np.max(sobel_abs))
    sobel_thresh = np.zeros_like(sobel_scale)
    sobel_thresh[(sobel_scale >= thresh[0]) & (sobel_scale <= thresh[1])] = 255
    return sobel_thresh


# Function to get gradient magnitude of an image
def gradientMag(img, k_size=3, thresh=(0,255)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
    sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
    sobelX_abs = np.absolute(sobelX)
    sobelY_abs = np.absolute(sobelY)
    sobel_mg = np.sqrt(np.square(sobelX_abs) + np.square(np.square(sobelY_abs)))
    sobel_scaled = np.uint8((sobel_mg * 255)/np.max(sobel_mg))
    sobel_thresh = np.zeros_like(sobel_scaled)
    sobel_thresh[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 255
    return sobel_thresh


# Function to get gradient direction of an image
def gradientDir(img, k_size=3, thresh=(0, np.pi/2)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
    sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
    sobelX_abs = np.absolute(sobelX)
    sobelY_abs = np.absolute(sobelY)
    sobel_dir = np.arctan2(sobelY_abs, sobelX_abs)
    sobel_thresh = np.zeros_like(sobel_dir)
    sobel_thresh[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 255
    return sobel_thresh


# Function to get color based threshold of an image
# The b channel in LAB is good for identifying staturated bright yellow lanes
# The l channel in HLS is good for identifying very bright white lanes
def colorThreshold(img, settings):
    # Convert color_space
    cspace = getattr(cv2, 'COLOR_RGB2'+settings['cspace'])
    img_cspace = cv2.cvtColor(img, cspace)
    img_ch = img_cspace[:,:,settings['channel']]

    # Apply CLAHE to improve contrast of image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_ch)

    # Get threshold
    thresh_low = settings['thresh'][0]
    thresh_high = settings['thresh'][1]

    # Apply threshold
    img_thresh = np.zeros_like(img_clahe)
    img_thresh[(img_clahe > thresh_low) & (img_clahe <= thresh_high)] = 255

    return  img_thresh


# Function to convert 3 channel image to binary image using gradient and color
# based thresholding
def convertToBinary(img, color_settings):
    #img_grad = gradientSobel(img, orient='x', thresh=thresh['sobel'])
    #img_grad_mag = gradientMag(img, thresh=thresh['mag'])
    #img_grad_dir = gradientDir(img, thresh=thresh['dir'])
    img_thresh = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
    for settings in color_settings:
        img_thresh += colorThreshold(img, settings)

    return img_thresh

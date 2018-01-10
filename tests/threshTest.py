import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Function to get color based threshold of an image
def colorThreshold(img, channel, thresh=(0, 255)):
    # Convert image to uint8
    if(img.dtype == 'float32'):
        img = np.uint8((img * 255)/np.max(img))

    img_ch = img

    # RGB
    if channel == 'rgb_r' or channel == 'rgb_g' or channel == 'rgb_b':
        if(channel == 'rgb_r'):
            img_ch = img[:,:,0]
        elif(channel == 'rgb_g'):
            img_ch = img[:,:,1]
        elif(channel == 'rgb_b'):
            img_ch = img[:,:,2]

    # HLS
    elif channel == 'hls_h' or channel == 'hls_l' or channel == 'hls_s':
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if(channel == 'hls_h'):
            img_ch = img_hls[:,:,0]
        elif(channel == 'hls_l'):
            img_ch = img_hls[:,:,1]
        elif(channel == 'hls_s'):
            img_ch = img_hls[:,:,2]

    # HSV
    elif channel == 'hsv_h' or channel == 'hsv_s' or channel == 'hsv_v':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if(channel == 'hsv_h'):
            img_ch = img_hsv[:,:,0]
        elif(channel == 'hsv_s'):
            img_ch = img_hsv[:,:,1]
        elif(channel == 'hsv_v'):
            img_ch = img_hsv[:,:,2]

    # LAB
    elif channel == 'lab_l' or channel == 'lab_a' or channel == 'lab_b':
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        if(channel == 'lab_l'):
            img_ch = img_lab[:,:,0]
        elif(channel == 'lab_a'):
            img_ch = img_lab[:,:,1]
        elif(channel == 'lab_b'):
            img_ch = img_lab[:,:,2]

    else:
        print("Invalid color channel!")
        quit()



    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_ch)

    img_thresh = np.zeros_like(img)
    img_thresh[(img_clahe > thresh[0]) & (img_clahe <= thresh[1])] = 255

    displayImages(img, img_thresh)

    return  img_thresh


# Function to get gradient of an image along x or y direction
def gradientSobel(img, orient='x', k_size=3, thresh=(0,255)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
    sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
    sobelX_abs = np.absolute(sobelX)
    sobelY_abs = np.absolute(sobelY)
    sobel_dir = np.arctan2(sobelY_abs, sobelX_abs)
    sobel_thresh = np.zeros_like(sobel_dir)
    sobel_thresh[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 255
    return sobel_thresh

def displayImages(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title('Image 1', fontsize=25)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Image 2', fontsize=25)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path',
                        type=str,
                        help='path to image file')
    parser.add_argument('channel',
                        type=str,
                        help='color channel [r, g, b, h, l, s]')
    parser.add_argument('t_low',
                        type=float,
                        help='lower threshold')
    parser.add_argument('t_high',
                        type=float,
                        help='higher threshold')
    args = parser.parse_args()

    img = mpimg.imread(args.img_path)

    img_thresh = colorThreshold(img, args.channel, (args.t_low, args.t_high))
    #img_thresh = gradientSobel(img, thresh=(args.t_low, args.t_high))
    #img_thresh = gradientMag(img, thresh=(args.t_low, args.t_high))
    #img_thresh = gradientDir(img, thresh=(args.t_low, args.t_high))
    #img_thresh = colorThreshold(img, args.channel, (args.t_low, args.t_high))

    #displayImages(img, img_thresh)

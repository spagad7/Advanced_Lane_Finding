#!/usr/bin/python3

import cv2
import pickle
import glob
import numpy as np

class Camera():
    def __init__(self, color_settings, img_settings, transform_settings):
        self.mtx = None
        self.dist = None
        self.M = None
        self.M_inv = None
        self.color_settings = color_settings
        self.img_warp_w = img_settings['img_warp_w']
        self.img_warp_h = img_settings['img_warp_h']
        self.img_w = img_settings['img_w']
        self.img_h = img_settings['img_h']
        self.trans_src = transform_settings['trans_src']
        self.trans_dst = transform_settings['trans_dst']

    # Function to calibrate camera
    def calibrate(self, dir_path, nx, ny):
        img_list = glob.glob(dir_path + "/*.jpg")
        objp = np.zeros((nx*ny, 3), dtype='float32')
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        img_pts = []
        obj_pts = []
        # Iterate through each calibration image
        for img_name in img_list:
            img = cv2.imread(img_name)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, corners = cv2.findChessboardCorners(img_gray,
                                                        (nx, ny), None)
            if retval == True:
                obj_pts.append(objp)
                img_pts.append(corners)
        # Calibrate Camera
        retval, self.mtx, self.dist, rvecs, tvecs = \
            cv2.calibrateCamera(obj_pts,
                                img_pts,
                                (img.shape[1], img.shape[0]),
                                None, None)


    # Function to save calibration data to file
    def saveCalibData(self, file_name):
        if(self.mtx == None or self.dist == None):
            print("Camera is not calibrated! Please calibrate the camera first")
            quit()
        f = open(file_name, "wb")
        pickle.dump(self.mtx, f)
        pickle.dump(self.dist, f)
        f.close()
        print("Saved calib data to:", file_name)

    # Function to read calibration data from file
    def loadCalibData(self, file_name):
        f = open(file_name, "rb")
        self.mtx = pickle.load(f)
        self.dist = pickle.load(f)
        f.close
        print("Read calib data from:", file_name)

    # Function to undistort image
    def undistortImg(self, img):
        img_undst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return img_undst

    # Function to calculate perspective transform
    def calcPerspectiveTransform(self):
        self.M = cv2.getPerspectiveTransform(self.trans_src, self.trans_dst)
        self.M_inv = np.linalg.inv(self.M)

    # Function to warp image
    def warpImage(self, img):
        if(self.M == None):
            print("Transform matrix not set! Please calculate perspective \
                    transform before warping an image.")
            quit()
        img_warp = cv2.warpPerspective(img,
                                        self.M,
                                        (self.img_warp_w, self.img_warp_h),
                                        flags=cv2.INTER_LINEAR)
        return img_warp

    # Function to unwarp image
    def unwarpImage(self, img):
        if(self.M_inv == None):
            print("Inverse transform matrix not set! Please calculate \
                    perspective transform before warping an image.")
            quit()
        img_unwarp = cv2.warpPerspective(img,
                                        self.M_inv,
                                        (self.img_w, self.img_h))
        return img_unwarp

    # Function to get gradient of an image along x or y direction
    def gradientSobel(self, img, orient='x', k_size=3, thresh=(0,255)):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if(orient == 'x'):
            sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
        elif(orient == 'y'):
            sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
        sobel_abs = np.absolute(sobel)
        sobel_scale = np.uint8((sobel_abs * 255)/np.max(sobel_abs))
        sobel_thresh = np.zeros_like(sobel_scale)
        sobel_thresh[(sobel_scale >= thresh[0])
                    & (sobel_scale <= thresh[1])] = 255
        return sobel_thresh

    # Function to get gradient magnitude of an image
    def gradientMag(self, img, k_size=3, thresh=(0,255)):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
        sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
        sobelX_abs = np.absolute(sobelX)
        sobelY_abs = np.absolute(sobelY)
        sobel_mg = np.sqrt(np.square(sobelX_abs)
                            + np.square(np.square(sobelY_abs)))
        sobel_scaled = np.uint8((sobel_mg * 255)/np.max(sobel_mg))
        sobel_thresh = np.zeros_like(sobel_scaled)
        sobel_thresh[(sobel_scaled >= thresh[0])
                    & (sobel_scaled <= thresh[1])] = 255
        return sobel_thresh

    # Function to get gradient direction of an image
    def gradientDir(self, img, k_size=3, thresh=(0, np.pi/2)):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
        sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
        sobelX_abs = np.absolute(sobelX)
        sobelY_abs = np.absolute(sobelY)
        sobel_dir = np.arctan2(sobelY_abs, sobelX_abs)
        sobel_thresh = np.zeros_like(sobel_dir)
        sobel_thresh[(sobel_dir >= thresh[0])
                    & (sobel_dir <= thresh[1])] = 255
        return sobel_thresh

    # Function to get color based threshold of an image
    # The b channel in LAB is good for identifying yellow lines
    # The l channel in HLS is good for identifying white lines
    def applyColorThreshold(self, img, settings):
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

        #if(settings['cspace'] == 'LAB'):
        #    img_nm = "images/img_lab.png"
        #else:
        #    img_nm = "images/img_hls.png"
        #cv2.imwrite(img_nm, img_thresh)
        return  img_thresh

    # Function to convert 3 channel image to binary image using gradient and color
    # based thresholding
    def convertToBinary(self, img):
        #img_grad = gradientSobel(img, orient='x', thresh=thresh['sobel'])
        #img_grad_mag = gradientMag(img, thresh=thresh['mag'])
        #img_grad_dir = gradientDir(img, thresh=thresh['dir'])
        img_thresh = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
        for settings in self.color_settings:
            img_thresh += self.applyColorThreshold(img, settings)
        #cv2.imwrite("images/img_thresh.png", img_thresh)
        return img_thresh

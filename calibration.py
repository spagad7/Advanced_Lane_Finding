#!/usr/bin/python3
import cv2
import glob
import pickle
import numpy as np


# Function to calibrate camera
def calibrateCamera(dir_path, nx, ny):
    img_list = glob.glob(dir_path + "/*.jpg")
    objp = np.zeros((nx*ny, 3), dtype='float32')
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    img_pts = []
    obj_pts = []
    # Iterate through each calibration image
    for img_name in img_list:
        img = cv2.imread(img_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)
        if retval == True:
            obj_pts.append(objp)
            img_pts.append(corners)
    # Calibrate Camera
    retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts,
                                        (img.shape[1], img.shape[0]),
                                        None, None)
    return mtx, dist


# Function to save calibration data to file
def saveCalibData(file_name, mtx, dist):
    f = open(file_name, "wb")
    pickle.dump(mtx, f)
    pickle.dump(dist, f)
    f.close()
    print("Saved calib data to:", file_name)


# Function to read calibration data from file
def loadCalibData(file_name):
    f = open(file_name, "rb")
    mtx = pickle.load(f)
    dist = pickle.load(f)
    f.close
    print("Read calib data from:", file_name)
    return mtx, dist

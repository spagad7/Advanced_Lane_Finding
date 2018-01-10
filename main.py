#!/usr/bin/python3
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from calibration import *
from threshold import *
from perspective import *
from lanes import *

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        type=str,
                        help='path to calibration file')
    parser.add_argument('-p',
                        type=str,
                        help='path to calibration images')
    parser.add_argument('-nx',
                        type=int,
                        help='number of inner corners along x direction')
    parser.add_argument('-ny',
                        type=int,
                        help='number of inner corners along y direction')
    parser.add_argument('-i',
                        type=str,
                        help='path to image file')
    parser.add_argument('-v',
                        type=str,
                        help='path to video file')
    args = parser.parse_args()

    # Error checking
    if(args.c == None and args.p == None):
        print("Insufficient arguments: either calib file or path to calib\
                images must be speicified!")
        quit()
    elif(args.p != None and (args.c == None or args.nx == None
        or args.ny == None)):
        print("Insufficient arguments: must specify calib file name, nx and ny")
        quit()

    if(args.i == None and args.v == None):
        print("Insufficient arguments: either image or video file must be\
                specified!")
        quit()

    # Calibrate Camera
    if args.p != None:
        mtx, dist = calibrateCamera(args.p, args.nx, args.ny)
        saveCalibData(args.c, mtx, dist)
    elif args.c != None and args.p == None:
        mtx, dist = loadCalibData(args.c)

    '''
    # Set threshold values
    thresh = {
              'sobel' : (65, 100),
              'mag' : (30, 100),
              'dir' : (1.1, 1.3),
              'hue_y' : (17, 30),
              'sat_y' : (160, 255),
              'light_y' : (200, 255),
              'hue_w' : (0, 0),
              'sat_w' : (0, 0),
              'light_w' : (0, 0),
              }
    '''
    color_settings = [
                        {'cspace':'LAB', 'channel':2, 'thresh':(150, 255)},
                        {'cspace':'HLS', 'channel':1, 'thresh':(210, 255)}
                     ]

    # Set transform coords
    trans_src = np.array([[535,32], [760,32],
                          [1120,200], [170,200]],
                           dtype='float32')
    offset = 10
    w = 200
    h = 400
    trans_dst = np.array([[offset, offset], [w+offset, offset],
                              [w+offset,h+offset], [offset,h+offset]],
                               dtype='float32')

    # This is temporary, will be removed after debugging
    # Read test image
    if args.i != None:
        img = cv2.imread(args.i)
        # Undistort image
        img_undst = cv2.undistort(img, mtx, dist, None, mtx)
        # Apply color and gradient threshold to the image
        img_bin = convertToBinary(img_undst, settings)
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=25)
        ax2.imshow(img_bin, cmap='gray')
        ax2.set_title('Binary Image', fontsize=25)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # Read video
    if args.v != None:
        print(args.v)
        video = VideoFileClip(args.v)
        cv2.namedWindow("Binary", flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Top-Down", flags=cv2.WINDOW_AUTOSIZE)

        #i = 0
        for frame in video.iter_frames():
            # Undistort frame
            frame_undst = cv2.undistort(frame, mtx, dist, None, mtx)
            # Crop the image
            top = 435
            bottom = 75
            h = frame_undst.shape[0]
            frame_crop = frame_undst[top:h-bottom, :]
            # Binarize frame
            frame_bin = convertToBinary(frame_crop, color_settings)
            #img_name = "videos/cropped2/img" + str(i) + ".png"
            #i = i + 1
            #cv2.imwrite(img_name, frame_bin)
            cv2.imshow("Binary", frame_bin)
            # Get top-down view of image
            frame_td = transformImage(frame_bin, trans_src, trans_dst)
            #cv2.imshow("Top-Down", frame_td)
            # Detect lanes
            img_lanes = detectLanes(frame_td)
            cv2.imshow("Top-Down", img_lanes)
            cv2.waitKey(1)

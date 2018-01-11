#!/usr/bin/python3
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from camera import *
from line import *


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

    # Configure camera settings
    color_settings = [
                        {'cspace':'LAB', 'channel':2, 'thresh':(150, 255)},
                        {'cspace':'HLS', 'channel':1, 'thresh':(205, 255)}
                     ]
    img_settings = {'img_w':1280, 'img_h':720,
                    'img_warp_w':250, 'img_warp_h':400}
    # Perspective transform coords
    trans_src = np.array([[565,455], [709,455],
                          [1100,655], [180,655]],
                           dtype='float32')
    offset = 10
    w = 200
    h = 400
    trans_dst = np.array([[offset, offset], [w+offset, offset],
                              [w+offset,h+offset], [offset,h+offset]],
                               dtype='float32')
    transform_settings = {'trans_src':trans_src, 'trans_dst':trans_dst}

    # Create camera object
    cam = Camera(color_settings, img_settings, transform_settings)

    # Calibrate Camera
    if args.p != None:
        cam.calibrate(args.p, args.nx, args.ny)
        cam.saveCalibData(args.c)
    elif args.c != None and args.p == None:
        cam.loadCalibData(args.c)

    # Calculate perspective transform
    cam.calcPerspectiveTransform()

    # Create object for left and right lines
    line_l = Line('left')
    line_r = Line('right')

    # Process video input
    if args.v != None:
        print(args.v)
        video = VideoFileClip(args.v)
        #cv2.namedWindow("Input", flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Top-Down", flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Output", flags=cv2.WINDOW_AUTOSIZE)

        for frame in video.iter_frames():
            # Undistort frame
            frame_undst = cam.undistortImg(frame)
            #cv2.imshow("Input", frame_undst)
            # Warp ROI to get top-down view
            frame_warp = cam.warpImage(frame_undst)
            cv2.imshow("Top-Down", frame_warp)
            # Convert warped frame to binary
            frame_bin = cam.convertToBinary(frame_warp)
            # Detect lanes
            frame_out = drawLane(cam, frame_undst, frame_bin, line_l, line_r)
            cv2.imshow("Output", frame_out)
            cv2.waitKey(1)

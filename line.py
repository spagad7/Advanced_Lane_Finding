#!/usr/bin/python3

import cv2
import numpy as np
from collections import deque

class Line():
    def __init__(self, side):
        self.side = side
        self.min_pix = 50
        self.n_windows = 7
        self.win_w = 20
        self.windows = []
        self.img_h = 0
        self.img_w = 0
        self.peak_cur = 0
        self.detected = False
        self.x_coords = None
        self.y_coords = None
        self.line_y = None
        self.line_x = None
        self.pts = np.array([])
        self.avg_x_list = deque()
        self.tol_x = 3


    # Function to find line in a binary image
    def findLine(self, img):
        self.img_h = img.shape[0]
        self.img_w = img.shape[1]
        win_h = self.img_h // self.n_windows

        # Find histogram of lower half of the image
        hist = np.sum(img[self.img_h//2:,:], axis=0)
        if self.side=='left':
            peak = np.argmax(hist[:self.img_w//2])
        elif self.side=='right':
            peak = np.argmax(hist[self.img_w//2:]) + self.img_w//2
        self.peak_cur = peak

        # Find nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = nonzero[0]
        nonzero_x = nonzero[1]

        line_inds = []
        # Line not detected in previous frame
        if self.detected == False:
            # Sliding Window algorithm
            for i in range(self.n_windows):
                # Calculate window boundaries only for first time
                win_low = self.img_h - (i)*win_h
                win_high = self.img_h - (i+1)*win_h
                win_left = self.peak_cur - self.win_w
                win_right = self.peak_cur + self.win_w
                self.windows.append({'win_low' : win_low,
                                     'win_high' : win_high,
                                     'win_left' : win_left,
                                     'win_right' : win_right})

                # Get indices of data points in nonzero_x and nonzero_y
                # which lie inside search window
                inds = ((nonzero_x >= win_left)
                        & (nonzero_x < win_right)
                        & (nonzero_y >= win_high)
                        & (nonzero_y < win_low)).nonzero()[0]
                line_inds.append(inds)

                # Update cur_peak_left and cur_peak_right
                if(inds.size > self.min_pix):
                    self.peak_cur = np.int(np.mean(nonzero_x[inds]))

            line_inds = np.concatenate(line_inds)

            # Use indices to get coordinates of pixels which lie
            # inside search windows
            self.x_coords = nonzero_x[line_inds]
            self.y_coords = nonzero_y[line_inds]

            # Line detected
            if (self.x_coords.size != 0 and self.y_coords.size != 0
                and self.lineDetected() == True):
                # x = ay^2 + by + c
                line_coeff = np.polyfit(self.y_coords, self.x_coords, 2)
                if self.line_y == None:
                    self.line_y = np.linspace(0, self.img_h-1, self.img_h)
                line_x_new = (line_coeff[0]*self.line_y**2
                                + line_coeff[1]*self.line_y
                                + line_coeff[2])
                # Line valid
                if self.lineValid(line_x_new) == True:
                    self.line_x = line_x_new
                else:
                    print(self.side +
                            " line invalid, using previous measurements")


            # Line not detected: use previous measurements
            else:
                print(self.side +
                        " Line not detected, using previous measurements")

        # TODO: Line detected in the previous frame
        # elif self.detected == True:


    # Function to check if line has been detected
    def lineDetected(self):
        if self.side == 'left':
            n = 5
        elif self.side == 'right':
            n = 10
        if len(np.unique(self.y_coords)) >= self.img_h/n:
            return True
        else:
            return False


    # Function to check validity of detected line
    def lineValid(self, line_x_new):
        n = 10
        popped = None
        avg_old = np.average(self.avg_x_list)
        if(len(self.avg_x_list) <= n):
            self.avg_x_list.append(np.average(line_x_new))
        else:
            popped = self.avg_x_list.popleft()
            self.avg_x_list.append(np.average(line_x_new))
        avg_new = np.average(self.avg_x_list)

        if abs(avg_old - avg_new) <= self.tol_x:
            return True
        else:
            #if(popped != None):
            #    self.avg_x_list.pop()
            #    self.avg_x_list.appendleft(popped)
            return False


    # Function to draw line
    def drawLine(self, img):
        if self.line_x != None and self.line_y != None:
            # Color pixels
            if self.side == 'left':
                img[self.y_coords, self.x_coords] = [255, 0, 0]
            elif self.side == 'right':
                img[self.y_coords, self.x_coords] = [0, 0, 255]

            # Draw rectangles
            for i in range(self.n_windows):
                cv2.rectangle(img,
                        (self.windows[i]['win_left'], self.windows[i]['win_low']),
                        (self.windows[i]['win_right'], self.windows[i]['win_high']),
                        (0, 255, 0), 1)
            self.windows = []

            # Draw line
            self.pts = np.array(np.transpose(np.vstack([self.line_x, \
                                                self.line_y]).astype('int32')))
            pts_rs = self.pts.reshape(-1, 1, 2)
            cv2.polylines(img, [pts_rs], False, [0, 255, 255], 1)
        return img


# End of class Line

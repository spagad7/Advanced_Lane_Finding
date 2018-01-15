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
        self.line_found = False
        self.x_coords = None
        self.y_coords = None
        self.line_y = None
        self.line_x = None
        self.x_m_per_px = 3.7/100
        self.y_m_per_px = 3/30
        self.pts = np.array([])
        self.prev_x_list = deque()
        self.prev_coeff = np.array([])
        self.prev_coeff_list = deque()
        self.prev_rad_list = deque()


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
        prev_n = 10

        # Line not detected in previous frame
        if self.line_found == False:
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

        # Line detected in previous frame
        elif self.line_found == True:
            margin = 25
            # Find indices of pixels which correspond to left lane
            line_x_low = (self.prev_coeff[0]*(nonzero_y**2)
                            + self.prev_coeff[1]*(nonzero_y)
                            + self.prev_coeff[2]) - margin
            line_x_high = (self.prev_coeff[0]*(nonzero_y**2)
                            + self.prev_coeff[1]*(nonzero_y)
                            + self.prev_coeff[2]) + margin
            inds = ((nonzero_x > line_x_low)
                    & (nonzero_x < line_x_high)).nonzero()[0]
            line_inds.append(inds)

        # Use indices to get coordinates of pixels which lie
        # inside search windows
        self.x_coords = nonzero_x[line_inds]
        self.y_coords = nonzero_y[line_inds]
        # Calculate y-coords of line
        if self.line_y == None:
            self.line_y = np.linspace(0, self.img_h-1, self.img_h)

        # Line detected
        if (self.x_coords.size != 0 and self.y_coords.size != 0
            and self.lineDetected() == True):
            # Calculate polynomial coefficients: x = ay^2 + by + c
            line_coeff = np.polyfit(self.y_coords, self.x_coords, 2)
            # Calculate the x-coords of the line
            line_x_new = (line_coeff[0]*self.line_y**2
                            + line_coeff[1]*self.line_y
                            + line_coeff[2])
            # Calculate average radius of curvature of line
            rad_curv_new = np.mean(
            ((1 + (2*line_coeff[0]*self.line_y + line_coeff[1])**2)**1.5)
            / np.absolute(2*line_coeff[0]))

            # Line valid
            if self.lineValid(line_x_new, rad_curv_new) == True:
                self.line_x = line_x_new
                self.prev_coeff = line_coeff
                self.line_found = True
                #print(self.side +
                #    ": updated prev_coeff: ",  self.prev_coeff)

            # Line invalid
            else:
                if(len(self.prev_coeff_list) != 0):
                    avg_coeff = np.mean(self.prev_coeff_list, axis=0)
                    self.line_x = (avg_coeff[0]*self.line_y**2
                                    + avg_coeff[1]*self.line_y
                                    + avg_coeff[2])
                    self.line_found = True
                    #print(self.side +
                    #    ": invalid, using avg_coeff: ",  avg_coeff)

            # Update prev_x_list
            if(len(self.prev_x_list) <= prev_n):
                self.prev_x_list.append(np.mean(line_x_new))
            else:
                self.prev_x_list.popleft()
                self.prev_x_list.append(np.mean(line_x_new))
            # Update prev_rad_list
            if(len(self.prev_rad_list) <= prev_n):
                self.prev_rad_list.append(rad_curv_new)
            else:
                self.prev_rad_list.popleft()
                self.prev_rad_list.append(rad_curv_new)
            # Update prev_coeff_list
            if(len(self.prev_coeff_list) <= prev_n):
                self.prev_coeff_list.append(line_coeff)
            else:
                self.prev_coeff_list.popleft()
                self.prev_coeff_list.append(line_coeff)

        # Line not detected: use previous measurements
        else:
            if(len(self.prev_coeff_list) != 0):
                avg_coeff = np.mean(self.prev_coeff_list, axis=0)
                self.line_x = (avg_coeff[0]*self.line_y**2
                                + avg_coeff[1]*self.line_y
                                + avg_coeff[2])
                self.line_found = False
                #print(self.side +
                #    ": not detected, using avg_coeff: ",  avg_coeff)


    # Function to check if line has been detected
    def lineDetected(self):
        # Set thresholds
        if self.side == 'left':
            n = 5
        elif self.side == 'right':
            n = 12
        # Check condition
        if len(np.unique(self.y_coords)) >= self.img_h/n:
            return True
        else:
            return False


    # Function to check validity of detected line
    def lineValid(self, line_x_new, rad_curv_new):
        tol_x = 10
        tol_rad = 1000
        if(len(self.prev_x_list) == 0 or len(self.prev_rad_list) == 0):
            return True

        avg_x = np.mean(self.prev_x_list)
        avg_rad = np.mean(self.prev_rad_list)
        cond_x = abs(avg_x - np.mean(line_x_new)) <= tol_x
        #if(cond_x == False):
            #print("cond_x = false: avg_x = ", avg_x, " line_x = ", np.mean(line_x_new))
        cond_rad = abs(avg_rad - rad_curv_new) <= tol_rad
        #if(cond_rad == False):
            #print("cond_rad = false: avg_rad = ", avg_rad, " rad_curv = ", rad_curv_new)
        return (cond_x and cond_rad)


    # Function to calculate radius of curvature of line
    def findRadCurv(self):
        rad_curv = 0
        if (self.line_y != None and self.line_x != None):
            y = np.max(self.line_y) * self.y_m_per_px
            line_coeff_sc = np.polyfit(self.line_y*self.y_m_per_px, self.line_x*self.x_m_per_px, 2)
            rad_curv = ((1 + (2*line_coeff_sc[0]*y + line_coeff_sc[1])**2)**1.5) \
                            / np.absolute(2 * line_coeff_sc[0])
            return rad_curv


    # Function to draw line
    def drawLine(self, img):
        if self.line_x != None and self.line_y != None:
            # Color pixels
            if self.side == 'left':
                img[self.y_coords, self.x_coords] = [255, 0, 0]
            elif self.side == 'right':
                img[self.y_coords, self.x_coords] = [0, 0, 255]

            # Draw rectangles
            if(self.line_found == False):
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

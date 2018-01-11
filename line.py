#!/usr/bin/python3

import cv2
import numpy as np

class Line():
    def __init__(self, side):
        self.side = side
        self.min_pix = 50
        self.n_windows = 9
        self.win_w = 25
        self.windows = []
        self.img_h = 0
        self.img_w = 0
        self.peak_cur = 0
        self.detected = False
        self.x = None
        self.y = None
        self.line_y = None
        self.line_x = None
        self.pts = None

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
        # if line not detected in previous frame
        if self.detected == False:
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

                # Get y-coord of points which lie inside the window
                inds = ((nonzero_x >= win_left)
                        & (nonzero_x < win_right)
                        & (nonzero_y >= win_high)
                        & (nonzero_y < win_low)).nonzero()[0]
                line_inds.append(inds)

                # Update cur_peak_left and cur_peak_right
                if(inds.size > self.min_pix):
                    self.peak_cur = np.int(np.mean(nonzero_x[inds]))

            line_inds = np.concatenate(line_inds)

        # TODO: write else condition
        # elif self.detected == True:

        # Get x and y coords of pixels lying on the line
        self.x = nonzero_x[line_inds]
        self.y = nonzero_y[line_inds]

        # TODO: change the if condition to condition for detecting a line
        if self.x.size != 0 and self.y.size != 0:
            # x = ay^2 + by + c
            line_coeff = np.polyfit(self.y, self.x, 2)
            if self.line_y == None:
                self.line_y = np.linspace(0, self.img_h-1, self.img_h)
            self.line_x = (line_coeff[0]*self.line_y**2
                            + line_coeff[1]*self.line_y
                            + line_coeff[2])

    # Function to draw line
    def drawLine(self, img):
        if self.line_x != None and self.line_y != None:
            # Color pixels
            if self.side == 'left':
                img[self.y, self.x] = [255, 0, 0]
            elif self.side == 'right':
                img[self.y, self.x] = [0, 0, 255]

            # Draw rectangles
            for i in range(self.n_windows):
                cv2.rectangle(img,
                        (self.windows[i]['win_left'], self.windows[i]['win_low']),
                        (self.windows[i]['win_right'], self.windows[i]['win_high']),
                        (0, 255, 0), 1)
            self.windows = []

            # Draw line
            self.pts = np.array(np.transpose(np.vstack([self.line_x, self.line_y]).astype('int32')))
            pts_rs = self.pts.reshape(-1, 1, 2)
            cv2.polylines(img, [pts_rs], False, [0, 255, 255], 1)
        return img


# Function to draw lane
def drawLane(cam, img_orig, img_bin, line_l, line_r):
    # Find lines
    line_l.findLine(img_bin)
    line_r.findLine(img_bin)
    # Draw lines
    img_zeros = np.zeros_like(img_bin).astype('uint8')
    img = np.dstack((img_zeros, img_zeros, img_zeros))*255
    img_lines = line_l.drawLine(img)
    img_lines = line_r.drawLine(img_lines)
    cv2.imshow("Lines", img_lines)

    # Create blank BGR image
    img_warp = np.zeros_like(img_orig).astype(np.uint8)
    # Arrange line points
    pts_lane = np.hstack((line_l.pts, line_r.pts))
    pts_lane = pts_lane.reshape(-1, 1, 2)
    # Draw lane
    cv2.fillPoly(img_warp, [pts_lane], (0, 255, 0))
    img_lanes = cam.unwarpImage(img_warp)
    img_result = cv2.addWeighted(img_orig, 1.0, img_lanes, 0.3, 0)
    return img_result

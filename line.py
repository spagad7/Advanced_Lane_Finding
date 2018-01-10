#!/usr/bin/python3

import cv2
import numpy as np

class Line():
    def __init__(self, camera):
        self.cam = camera
        self.min_pix = 50
        self.n_windows = 9
        self.win_width = 25
        self.plot_x_left = None
        self.plot_x_right = None
        self.plot_y = None
        self.pts_left = None
        self.pts_right = None

    # Function to detect lanes
    def fit(self, img):
        lanes_found = False
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_out = np.dstack((img, img, img))*255

        # Get indices of non-zero pixels
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        self.min_pix = 50
        left_lane_inds = []
        right_lane_inds = []

        if(lanes_found == False):
            # Configure window
            win_height = img_h//self.n_windows

            # Get histogram of lower half of the image
            hist = np.sum(img[img_h//2:,:], axis=0)

            # Get the index of peak in left and right half
            midpoint = hist.shape[0]//2
            peak_left = np.argmax(hist[:midpoint])
            peak_right = np.argmax(hist[midpoint:]) + midpoint
            cur_peak_left = peak_left
            cur_peak_right = peak_right

            # Find lane in each window
            for window in range(self.n_windows):
                # Define boundary for left and right windows
                win_bottom = img_h - (window+1) * win_height
                win_top = img_h - window * win_height
                win_left_left = cur_peak_left - self.win_width
                win_left_right = cur_peak_left + self.win_width
                win_right_left = cur_peak_right - self.win_width
                win_right_right = cur_peak_right + self.win_width

                # Draw left window
                cv2.rectangle(img_out,
                              (win_left_left, win_bottom),
                              (win_left_right, win_top),
                              (0, 255, 0), 1)
                # Draw right window
                cv2.rectangle(img_out,
                              (win_right_left, win_bottom),
                              (win_right_right, win_top),
                              (0, 255, 0), 1)

                # Find nonzero pixels which lie inside left and right windows
                left_inds = ((nonzero_x >= win_left_left) &
                                (nonzero_x < win_left_right) &
                                (nonzero_y >= win_bottom) &
                                (nonzero_y < win_top)).nonzero()[0]

                right_inds = ((nonzero_x >= win_right_left) &
                              (nonzero_x < win_right_right) &
                              (nonzero_y >= win_bottom) &
                              (nonzero_y < win_top)).nonzero()[0]

                left_lane_inds.append(left_inds)
                right_lane_inds.append(right_inds)

                # Update cur_peak_left and cur_peak_right
                if(len(left_inds) > self.min_pix):
                    cur_peak_left = np.int(np.mean(nonzero_x[left_inds]))
                if(len(right_inds) > self.min_pix):
                    cur_peak_right = np.int(np.mean(nonzero_x[right_inds]))

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)


        elif(lanes_found == True):
            margin = 25
            # Find indices of pixels which correspond to left lane
            left_x_low = (left_lane[0]*(nonzero_y**2)
                            + left_lane[1]*(nonzero_y)
                            + left_lane[2] - margin)
            left_x_high = (left_lane[0]*(nonzero_y**2)
                            + left_lane[1]*(nonzero_y)
                            + left_lane[2] + margin)
            left_inds =((nonzero_x > left_x_low)
                        & (nonzero_x < left_x_high)).nonzero()[0]

            # Find indices of pixels which correspond to right lane
            right_x_low = (right_lane[0]*(nonzero_y**2)
                            + right_lane[1]*(nonzero_y)
                            + right_lane[2] - margin)
            right_x_high = (right_lane[0]*(nonzero_y**2)
                            + right_lane[1]*nonzero_y
                            + right_lane[2] + margin)
            right_inds = ((nonzero_x > right_x_low)
                            & (nonzero_x < right_x_high)).nonzero()[0]

            left_lane_inds.append(left_inds)
            right_lane_inds.append(right_inds)


        # Get indices of all the pixels which correspond to left and right lanes
        left_lane_x = nonzero_x[left_lane_inds]
        left_lane_y = nonzero_y[left_lane_inds]
        right_lane_x = nonzero_x[right_lane_inds]
        right_lane_y = nonzero_y[right_lane_inds]

        if(left_lane_x.size != 0 and left_lane_y.size != 0 and
        right_lane_x.size != 0 and right_lane_y.size != 0):
            # Get coefficients of the curve
            left_lane = np.polyfit(left_lane_y, left_lane_x, 2)
            right_lane = np.polyfit(right_lane_y, right_lane_x, 2)
            lanes_found = False
            # Generate y values
            if(self.plot_y == None):
                self.plot_y = np.linspace(0, img_h-1, img_h)
            # Calculate x values for left and right lane
            self.plot_x_left = (left_lane[0]*(self.plot_y**2)
                                + left_lane[1]*self.plot_y
                                + left_lane[2])
            self.plot_x_right = (right_lane[0]*(self.plot_y**2)
                                + right_lane[1]*self.plot_y
                                + right_lane[2])

            img_out[nonzero_y[left_lane_inds],
                    nonzero_x[left_lane_inds]] = [255, 0, 0]
            img_out[nonzero_y[right_lane_inds],
                    nonzero_x[right_lane_inds]] = [0, 0, 255]

            # Draw left line
            self.pts_left = np.array(np.transpose(np.vstack([self.plot_x_left,\
                                                self.plot_y]).astype('int32')))
            pts_left_rs = self.pts_left.reshape(-1, 1, 2)
            cv2.polylines(img_out, [pts_left_rs], False, (0, 255, 255), 4)

            # Draw right line
            self.pts_right = np.array(np.transpose(np.vstack([self.plot_x_right,\
                                                self.plot_y]).astype('int32')))
            pts_right_rs = self.pts_right.reshape(-1, 1, 2)
            cv2.polylines(img_out, [pts_right_rs], False, (0, 255, 255), 4)
        else:
            lanes_found = False
        cv2.imshow("Lane Lines", img_out)

    # Function to draw lane
    def drawLane(self, img):
        # Create blank BGR image
        img_warp = np.zeros_like(img).astype(np.uint8)
        # Arrange line points
        pts_lane = np.hstack((self.pts_left, self.pts_right))
        pts_lane = pts_lane.reshape(-1, 1, 2)
        # Draw lane
        cv2.fillPoly(img_warp, [pts_lane], (0, 255, 0))
        img_lanes = self.cam.unwarpImage(img_warp)
        img_result = cv2.addWeighted(img, 1.0, img_lanes, 0.3, 0)
        return img_result

import cv2
import argparse
import numpy as np

def getColor(event, x, y, flags, userdata):
    if(event == cv2.EVENT_LBUTTONDOWN):
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_h = img_hls[:,:,0]
        img_l = img_hls[:,:,1]
        img_s = img_hls[:,:,2]
        h_val = img_h[y][x]
        s_val = img_s[y][x]
        l_val = img_l[y][x]
        #print("x = ", x, " y = ", y)
        print("Hue = ", h_val)
        print("Saturation = ", s_val)
        print("Lightness = ", l_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('i',
                        type=str,
                        help='path to image file')
    args = parser.parse_args()

    img = cv2.imread(args.i)

    cv2.namedWindow("Color Picker", flags=cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Color Picker", getColor, 0)
    cv2.imshow("Color Picker", img)
    cv2.waitKey(0)

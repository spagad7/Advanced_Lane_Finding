import cv2
import numpy as np

def transformImage(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img, M, (250, 400), flags=cv2.INTER_LINEAR)
    return img_warp

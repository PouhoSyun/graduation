import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import cv2, torch
import numpy as np
from project.utils import *
        
def test_fine_split(pic_no):
    win = cv2.namedWindow("win", cv2.WINDOW_FREERATIO)

    picset = Frame_Dataset("Flowers", cv2.IMREAD_GRAYSCALE, 500)
    pic = np.array(picset[pic_no])
    split = fine_split16(pic)
    stack = hvstack16(split)

    show = np.hstack((pic, stack)).astype(np.uint8)
    cv2.imshow("win", show)
    cv2.waitKey(0)

def test_event_voxel():
    dataset = DAVIS_Dataset("Indoor4", 0.02, cv2.IMREAD_GRAYSCALE, 1)
    while(True):
        dataset.__show__(int(input("Index to preview: ")))

if __name__ == "__main__":
    # test_fine_split(1)
    test_event_voxel()
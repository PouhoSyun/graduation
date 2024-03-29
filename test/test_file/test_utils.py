import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import cv2, torch
import numpy as np
from project.utils import *
        
def test_fine_split(pic_no):
    _ = cv2.namedWindow("win", cv2.WINDOW_FREERATIO)

    picset = Frame_Dataset("Flowers", cv2.IMREAD_GRAYSCALE, 500)
    pic = np.array(picset[pic_no])
    split = fine_split(pic)
    stack = hvstack(split)
    fine = fine_stack(split)

    show = np.hstack((pic, stack, fine)).astype(np.uint8)
    cv2.imshow("win", show)
    cv2.waitKey(0)

def test_event_voxel(pic_no):
    _ = cv2.namedWindow("win", cv2.WINDOW_AUTOSIZE)

    voxel = np.array(pack_event_stream("Indoor4", 200, 0.02, False)[pic_no])
    # voxel = cv2.copyMakeBorder(voxel, 700, 700, 700, 700, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    frame = np.array(load_frame_png("Indoor4", pic_no, cv2.IMREAD_GRAYSCALE, False))
    frame_dst = np.array(load_frame_png("Indoor4", pic_no + 1, cv2.IMREAD_GRAYSCALE, False))
    # show = np.hstack([frame, voxel, frame_dst]).astype(np.uint8)
    show = voxel.astype(np.uint8)
    cv2.imshow("win", show)
    cv2.waitKey(0)

if __name__ == "__main__":
    # test_fine_split(1)
    test_event_voxel(1)
import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import mat73, cv2, math
import numpy as np
import torch
from project.utils import *
        
if __name__ == '__main__':
    win = cv2.namedWindow("win", cv2.WINDOW_FREERATIO)

    pic = "test\dataset\Flowers\image_00001.jpg"
    split = fine_split16(pic)
    stack = hvstack16(split)

    show = np.hstack((pic, stack)).astype(np.uint8)
    cv2.imshow("win", show)
    cv2.waitKey(0)

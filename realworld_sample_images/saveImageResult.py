from PIL import Image
import os, os.path
import cv2
import numpy as np


# path = '/home/etri/DB/DB_result/stuttgart_00/'
# path2 = '/home/etri/DB/DB_result/stuttgart_00_result/'
# path3 = '/home/etri/DB/DB_result/stuttgart_00_combine/'


path = '/home/etri/DB/1_original/'
path2 = '/home/etri/DB/2_result/'
path3 = '/home/etri/DB/3_combine/'


for name in os.listdir(path2):

    print('name:', path2+name)
    back = cv2.imread(path+name, cv2.IMREAD_COLOR)
    fore = cv2.imread(path2+name, cv2.IMREAD_COLOR)

    # print(back.shape)
    # print(fore.shape)

    alpha=0.5
    dst = cv2.addWeighted(back, alpha, fore, (1-alpha), 0 )

    # cv2.imshow('dst', dst)
    cv2.imwrite(path3+name, dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



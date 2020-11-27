import cv2
import numpy as np
import numpy
import sys
import pandas as pd
# from treelib import Node, Tree
from Matrixlists import *
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
#ddddddddddddddddddddddd
pd.set_option('display.max_columns',10)
img = cv2.imread(r'D:\Eng\Semsters\NEW\10th\Autonomus/37.png', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
map = np.array(thresh1)
# cv2.imshow('image', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()
###########################################################
num_rows = len(map) #num of rows of the map(image)=400
num_cols = len(map[0]) #num of cols of the map(image)=1000
dublicate = np.zeros((num_rows, num_cols)) # another map for output path
dublicate1 = np.zeros((num_rows,num_cols,2),dtype=int) # another map for output path


#######################################################################
obst = []
for i in range(400):
    for j in range(1000):
        if map[i][j] == 0:
            obst.append([i/100,j/100])



print(obst)
print(len(obst))

# dublicate2 = np.zeros((num_rows, num_cols)) # another map for output path


#
# for i in Path:
#     colored_map[i[0]][i[1]] = (0, 0, 255)
# plt.imshow(colored_map)
# plt.xlabel("x axis label")
# plt.ylabel("y axis label")
# plt.show()
# cv2.waitKey(0)

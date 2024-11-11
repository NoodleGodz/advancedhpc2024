import math
import numpy as np
import matplotlib.pyplot as plt
import time
import numba
from numba import cuda


w = 17
H,W = 16+2*w,16+2*w
image = np.zeros((H,W),np.uint)
print(image.shape)

for i in range(16):
    for j in range(16):
        a,b = i+w,j+w
        image[a,b]=1
        image[a+w,b]=1
        image[a,b+w]=1
        image[a,b-w]=1
        image[a-w,b]=1
        image[a+w,b+w]=1
        image[a-w,b+w]=1
        image[a+w,b-w]=1
        image[a-w,b-w]=1

plt.title(f"Window size = {w}")
plt.imshow(image,cmap='gist_grey')
plt.grid(True)
plt.show()

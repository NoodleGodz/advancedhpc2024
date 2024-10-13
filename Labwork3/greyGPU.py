import numba
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

hostInp=plt.imread("images/image1.jpg")

print(hostInp.dtype)
(H,W,C) = hostInp.shape

pixelcount= H*W
hostInp=hostInp.reshape((pixelcount,3))
devOut = cuda.device_array((pixelcount, C), np.uint8)



hostOut=np.zeros((H,W,C),np.uint8)

devInp=cuda.to_device(hostInp)

@cuda.jit

def grayscale(src, dst):

    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g


blockSize = 64
gridSize = pixelcount // blockSize
grayscale[gridSize, blockSize](devInp, devOut)
hostOut = devOut.copy_to_host()

hostOut.reshape((H,W,C))

plt.imshow(hostOut)


plt.show()
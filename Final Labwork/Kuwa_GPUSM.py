import math
import numpy as np
import matplotlib.pyplot as plt
import time
import numba
from numba import cuda

@cuda.jit
def rgb_to_v_kernel(input_image, v_channel):
    x, y = cuda.grid(2)
    (H,W,_) = input_image.shape

    if y < H and x < W:
        r = input_image[y, x, 0] / 255.0
        g = input_image[y, x, 1] / 255.0
        b = input_image[y, x, 2] / 255.0

        v_channel[y, x] = max(r, g, b)

@cuda.jit(device=True)
def calculate_std(v_channel, start_x, start_y, w):
    #print(start_x, " ", start_y)
    sum_v, sum_sq_v, count = 0.0, 0.0, 0
    for i in range(0, w + 1):
        for j in range(0, w + 1):
            nx, ny = start_x + i, start_y + j
            sum_v += v_channel[ny, nx]
            count += 1
    mean = sum_v / count
    for i in range(0, w + 1):
        for j in range(0, w + 1):
            nx, ny = start_x + i, start_y + j
            v = v_channel[ny, nx]
            diff = v - mean
            sum_sq_v += diff * diff
    std = math.sqrt( sum_sq_v/ count)
    return std


@cuda.jit(device=True)
def calculate_mean_color(window, w):
    r_sum, g_sum, b_sum = 0, 0, 0
    count = (w + 1) ** 2
    for i in range(w + 1):
        for j in range(w + 1):
            r_sum += window[i, j, 0]
            g_sum += window[i, j, 1]
            b_sum += window[i, j, 2]
    return r_sum // count, g_sum // count, b_sum // count

@cuda.jit
def load_share_mem(shared_image,shared_v,image,v,w):
    x, y = cuda.grid(2)
    x, y = x+w,y+w
    local_x, local_y = cuda.threadIdx.x, cuda.threadIdx.y 
    H, W, _ = image.shape
    if x < W-w and y< H-w:
        shared_v[local_y, local_x] = v[y, x]
        shared_v[local_y-w, local_x-w] = v[y-w, x-w]
        shared_v[local_y+w, local_x-w] = v[y+w, x-w]
        shared_v[local_y-w, local_x+w] = v[y-w, x+w]
        shared_v[local_y+w, local_x+w] = v[y+w, x+w]
        shared_v[local_y-w, local_x] = v[y-w, x]
        shared_v[local_y+w, local_x] = v[y+w, x]
        shared_v[local_y, local_x-w] = v[y, x-w]
        shared_v[local_y, local_x+w] = v[y, x+w]
        for i in range(3):
            shared_image[local_y, local_x,i] = image[y, x,i]
            shared_image[local_y-w, local_x-w,i] = image[y-w, x-w,i]
            shared_image[local_y+w, local_x-w,i] = image[y+w, x-w,i]
            shared_image[local_y-w, local_x+w,i] = image[y-w, x+w,i]
            shared_image[local_y+w, local_x+w,i] = image[y+w, x+w,i]
            shared_image[local_y-w, local_x,i] = image[y-w, x,i]
            shared_image[local_y+w, local_x,i] = image[y+w, x,i]
            shared_image[local_y, local_x-w,i] = image[y, x-w,i]
            shared_image[local_y, local_x+w,i] = image[y, x+w,i]
            
    cuda.syncthreads()


@cuda.jit
def kuwahara_kernel(input_image, output_image, v_channel, w):
    # need to load (16+2w,16+2w) here using (16,16) threads
    x, y = cuda.grid(2)
    x, y = x+w,y+w
    local_x, local_y = cuda.threadIdx.x, cuda.threadIdx.y 
    #local_x, local_y = local_x + w, local_y + w
    
    H, W, _ = input_image.shape
    #Hardcode the shared array alloc due to cuda, can't use variable, if need bigger w change here, set at 11 as the biggest test
    shared_image = cuda.shared.array((16+2*11,16+2*11, 3),numba.uint8)
    shared_v_channel = cuda.shared.array((16+2*11,16+2*11),numba.float32)
    load_share_mem(shared_image,shared_v_channel,input_image,v_channel,w)
    cuda.syncthreads()
    
    if x < W-w and y < H-w:
        stds = cuda.local.array((4), numba.float32)
        stds[0] = calculate_std(shared_v_channel, local_x-w, local_y-w, w)
        stds[1] = calculate_std(shared_v_channel, local_x, local_y-w, w)       
        stds[2] = calculate_std(shared_v_channel, local_x-w, local_y, w)
        stds[3] = calculate_std(shared_v_channel, local_x, local_y, w)
        cuda.syncthreads()
        #print(stds[0] , "/", stds[1] , "/", stds[2] , "/", stds[3])
        

        min_index = 0
        for i in range(1, 4):
            if stds[i] < stds[min_index]:
                min_index = i
        window = cuda.local.array((12, 12, 3), numba.uint8)
        start_x = local_x-w if min_index in [0,2] else local_x
        start_y = local_y-w if min_index in [0,1] else local_y
        for i in range(w + 1):
            for j in range(w + 1):
                nx = start_x + i
                ny = start_y + j
                window[i, j, 0] = shared_image[ny, nx, 0]
                window[i, j, 1] = shared_image[ny, nx, 1]
                window[i, j, 2] = shared_image[ny, nx, 2]

        r, g, b = calculate_mean_color(window, w)
        
        output_image[y-w, x-w, 0] = r
        output_image[y-w, x-w, 1] = g
        output_image[y-w, x-w, 2] = b


def kuwahara_filter(image,w):
    (H,W,C) = image.shape

    output = np.zeros((H,W,C), np.uint8)
    padded_image = np.pad(image, ((w, w), (w, w), (0, 0)), mode='reflect')
    (H,W,C) = padded_image.shape
    v_channel = np.zeros((H, W), dtype=np.float32)

    d_input_image = cuda.to_device(padded_image)
    d_v_channel = cuda.to_device(v_channel)
    d_output = cuda.to_device(output)
    nb_threads = (16,16)
    nb_blocks_1 = (H + nb_threads[0] - 1) // nb_threads[0]
    nb_blocks_2 = (W + nb_threads[1] - 1) // nb_threads[1]

    nb_blocks = (nb_blocks_2,nb_blocks_1)
    #print(f"Number of blocks: {nb_blocks}")
    #print(f"Number threads per blocks: {nb_threads}")

    rgb_to_v_kernel[nb_blocks, nb_threads](d_input_image, d_v_channel)
    cuda.synchronize()
    kuwahara_kernel[nb_blocks, nb_threads](d_input_image, d_output, d_v_channel, w)
    cuda.synchronize()
    return d_output.copy_to_host()


w = 3
path = "Kuwa.jpg"

input_image = plt.imread(path)
for i in range(10):
    output_image = kuwahara_filter(input_image,w)
for i in range(3,12,2):
    w = i
    path = "Kuwa.jpg"

    input_image = plt.imread(path)
    print(input_image.dtype)
    print(input_image.shape)

    t1 = time.time()
    output_image = kuwahara_filter(input_image,w)
    t2 = time.time()
    print(f"Time taken for Kuwahara GPU no SM{w} filter: {t2 - t1} seconds")

    plt.imsave(f"Kuwa_GPUSM_(w={w}).jpg",output_image)
plt.imshow(output_image)

plt.tight_layout()

plt.show()
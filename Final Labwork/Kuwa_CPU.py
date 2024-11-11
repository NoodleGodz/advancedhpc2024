import numpy as np
import matplotlib.pyplot as plt
import time

def rgb_to_v(image):
    (H,W,_) = image.shape
    output = np.zeros((H,W), np.float32)

    for i in range(H):
        for j in range(W):
            r_v = image[i, j, 0]/255
            g_v = image[i, j, 1]/255
            b_v = image[i, j, 2]/255
            
            output[i, j] = max(r_v, g_v, b_v)

    return output

def get_windows(image, i, j, w):
    i,j = i+w,j+w
    #print(f"i={i},j={j}")
    windows = np.zeros((4, w+1, w+1, 3), np.uint8)
    windows[0] = image[i-w:i+1, j-w:j+1, :]
    windows[1] = image[i:i+w+1, j-w:j+1, :]
    windows[2] = image[i-w:i+1, j:j+w+1, :]
    windows[3] = image[i:i+w+1, j:j+w+1, :]
    
    return windows

def get_std_v(v_v,i,j,w):
    i,j = i+w,j+w
    #print(f"i={i},j={j}")
    v_list = np.zeros((4), np.float32)
    v_list[0] = np.std(v_v[i-w:i+1, j-w:j+1])
    v_list[1] = np.std(v_v[i:i+w+1, j-w:j+1])
    v_list[2] = np.std(v_v[i-w:i+1, j:j+w+1])
    v_list[3] = np.std(v_v[i:i+w+1, j:j+w+1])
    return v_list

def get_mean(window):
    r_v = int(np.mean(window[:,:,0]))
    g_v = int(np.mean(window[:,:,1]))
    b_v = int(np.mean(window[:,:,2]))
    return r_v,g_v,b_v



def kuwahara_cpu(image,w):
    (H,W,C) = image.shape
    output = np.zeros((H,W,C), np.uint8)

    padded_image = np.pad(image, ((w, w), (w, w), (0, 0)), mode='reflect')
    v_v = rgb_to_v(padded_image)


    for i in range(H):
        for j in range(W):
            #print(f"i={i},j={j}")
            wins = get_windows(padded_image,i,j,w)
            v_list = get_std_v(v_v,i,j,w)
            output[i,j]= get_mean(wins[np.argmin(v_list)])
    return output

w = 11
path = "Kuwa.jpg"

input_image = plt.imread(path)
print(input_image.dtype)
print(input_image.shape)



t1 = time.time()
output_image = kuwahara_cpu(input_image,w)
t2 = time.time()
print(f"Time taken for Kuwahara CPU {w} filter: {t2 - t1:.3f} seconds")
plt.imsave(f"Kuwa_CPU_(w={w}).jpg",output_image)

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_image)
plt.axis("off")


plt.subplot(1, 2, 2)
plt.title(f"Kuwahara Filter Output (w={w})")
plt.imshow(output_image)
plt.axis("off")
plt.tight_layout()

plt.show()


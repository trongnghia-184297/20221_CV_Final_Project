import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log
from statistics import mean


class Denoise:
    def __init__(self, path_in):
        self.path_in = path_in
        self.path_out = 'denoise'
        # pass

    def create_folder(self, path_out):
        if not os.path.exists(path_out):
            os.mkdir(path_out)

    def local_maximum(self, spec,win_size,thresh):
        wid, hei = spec.shape
        wid_loop_times = int(wid/win_size)
        hei_loop_times = int(hei/win_size)
        global_max = np.max(spec)
        max_pos = []
        for i in range(wid_loop_times):
            for j in range(hei_loop_times):
                window = spec[win_size*i:win_size*(i+1), win_size*j:win_size*(j+1)]
                local_max = np.max(window)
                if (local_max > (global_max*thresh) and local_max < global_max):
                    ind = np.unravel_index(np.argmax(window, axis=None), window.shape)
                    max_pos.append((5*i+ind[0], 5*j+ind[1]))
        return max_pos

    def denoise(self):

        # Create path_out folder
        self.create_folder(self.path_out)
        
        # Get path_out for image to save
        name_list = self.path_in.split("/")
        name_png = name_list[-1]
        path_out_png = self.path_out + "/" + name_png
        
        # Reading input image
        img = cv2.imread(self.path_in)
        # cv2.imshow("Original image",img)

        # Filter noise and convert to gray image
        median = cv2.medianBlur(img,5)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        
        # Using gamma correction
        gamma = log(img.mean())/log(128)
        if gamma < 0.5:
            gamma = 0.1
        # print(gamma)
        gamma_corrected = np.array(255*(gray / 255) ** gamma, dtype = 'uint8')

        # convert image to floats and do dft saving as complex output
        dft = cv2.dft(np.float32(gamma_corrected), flags = cv2.DFT_COMPLEX_OUTPUT)
        # apply shift of origin from upper left corner to center of image
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        max_pos = self.local_maximum(magnitude_spectrum, 5, 0.9)

        # remove sin noise
        max_pos = self.local_maximum(magnitude_spectrum, 5, 0.9)
        for pos in max_pos:
            # remove sin noise
            dft_shift[pos] = 0

        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)

        # do idft saving as complex output
        img_back = cv2.idft(back_ishift)

        # combine complex components into original image again
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # dimensions = img_back.shape
        # height = img.shape[0]
        # width = img.shape[1]
        # channels = img.shape[2]
        
        img_back[:,-1] = img_back[:,-2]
        img_back[-1,:] = img_back[-2,:]
        
        plt.imsave(path_out_png, img_back, cmap='gray')
        

denoise_test = Denoise("imgs/noise_periodic.png")
# denoise_test = Denoise("imgs/noise_pepper.png")
# denoise_test = Denoise("imgs/noise_connected.png")
# denoise_test = Denoise("imgs/normal.png")
denoise_test.denoise()



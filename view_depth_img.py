import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

if os.path.isfile('real/background_heightmap.depth.png'):
    import cv2
        # load depth image saved in 1e-5 meter increments 
        # see logger.py save_heightmaps() and trainer.py load_sample() 
        # for the corresponding save and load functions
    background_heightmap = np.array(cv2.imread('real/background_heightmap.depth.png', cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 100000

    plt.imshow(background_heightmap)
    plt.show()

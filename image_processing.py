import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io, measure, color
from skimage.morphology import medial_axis, remove_small_objects, closing, disk
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label
import matplotlib.patches as patches


class ImageProcessing:
    
    def __init__(self, image_path, bbox):
        self.img = io.imread(image_path)
        self.bbox = bbox
    
    def trim_image(self):
        self.trimmed_image = self.img[615:3835, 410:3650]
        self.trimmed_image = self.trimmed_image[450:4000, 250:3750]
        
        plt.imsave('trim_class.png', self.trimmed_image)
        
        self.trim_image = 'trim_class.png'
        
        return self.trim_image
        
        
    def mask_image(self):
        """
        This function takes in an image, converts it from RGB to HSV so that it can focus on the red space hue
        """
        hsv = cv2.cvtColor(self.trimmed_image, cv2.COLOR_RGB2HSV)
        self.mask = cv2.inRange(hsv,(130, 50, 20), (180, 255, 255)) # Red
        
        plt.imsave('mask_image.png', self.mask)
        
        self.masks = 'mask_image.png'
        
        return self.mask
    
    def labeled_image(self):
        
        self.label_image = measure.label(self.mask)
        

        plt.imsave('labeled_image.png', self.label_image)
        
        #self.label_image = 'labeled_image.png'

        
        return self.label_image
    
    def skeletonize_image(self):
        # Read image
        blobs_color = io.imread(self.masks)
        
        # Drop alpha channel if present
        if blobs_color.shape[-1] == 4:
            blobs_color = blobs_color[:, :, :3]
        
        # Convert to grayscale
        blobs_gray = color.rgb2gray(blobs_color)

        # Smooth to reduce noise before thresholding
        blobs_gray_smooth = gaussian(blobs_gray, sigma=1.0)

        # Binarize with Otsu
        thresh = threshold_otsu(blobs_gray_smooth)
        blobs_binary = blobs_gray_smooth > thresh
        
        # Remove small objects (noise) and # Morphological closing to fill small holes
        blobs_clean = remove_small_objects(label(blobs_binary), min_size=500)
        blobs_clean = closing(blobs_clean, disk(3))
        
        # Medial axis skeleton
        skel, distance = medial_axis(blobs_clean, return_distance=True)
        
        io.imsave('skeleton_image.png', (skel * 255).astype('uint8'))
        
        self.skeleton_img = 'skeleton_image.png'
        
        return self.skeleton_img
        

    def process(self):
        print("\n--- Running Image Processing Pipeline ---")
        trimmed_path = self.trim_image()
        mask_path = self.mask_image()
        label_img = self.labeled_image()  # returns label matrix (no save)
        skeleton_path = self.skeletonize_image()
        print("\n Image Processing pipeline completed successfully.")




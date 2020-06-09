import numpy as np
import scipy.ndimage as ndi
import skimage.feature
import matplotlib.pyplot as plt
import skimage.io as io
import skimage
try:
    import seaborn
except ImportError:
    pass
try:
    import yellowbrick
except ImportError:
    pass

urls = ["E:\Rasp\Source_data\c8\IMG_9004.JPG"]

def get_gradient():
    images = io.imread_collection(urls,conserve_memory=True,plugin=None)
    for image in images:
        fd, hog_image = skimage.feature.hog(image,
                                  orientations=9,
                                  pixels_per_cell=(8,8),
                                  cells_per_block=(3,3),
                                  block_norm='L2',
                                  visualize=True,
                                  transform_sqrt=True,
                                  feature_vector=True,
                                  multichannel=True)
    yield fd, hog_image



# =====================================================================

if __name__ == '__main__':
    image = skimage.io.imread("E:\Rasp\Source_data\c8\IMG_9004.JPG")
    fd, hog_image = skimage.feature.hog(image,
                              orientations=8 ,
                              pixels_per_cell=(8,8),
                              cells_per_block=(3,3),
                              block_norm='L2',
                              visualize=True,
                              transform_sqrt=True,
                              feature_vector=True,
                              multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    

    # Rescale histogram for better display
    hog_image_rescaled = skimage.exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    plt.close()
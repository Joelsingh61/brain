from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_file):
    img = image.load_img(img_file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imageio
from PIL import Image
import os
import cv2


def model_predict(input_url):
    # Load TFLite model and allocate tensors.
    #model_lite_path = os.path.join('topartist_rn50.tflite')
    model_lite_path = tf.keras.utils.get_file('lite_model_deepartist_rn50.tflite', 'https://www.dropbox.com/s/bodh4db0j1bz8b4/lite_model_deepartist_rn50.tflite?dl=1')
    print("Imported the model")
    interpreter = tf.lite.Interpreter(model_path=model_lite_path)
    interpreter.allocate_tensors()

	# Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    artists = {
			0: 'Vincent van Gogh',
			1: 'Edgar Degas',
			2: 'Pablo Picasso',
			3: 'Pierre-Auguste Renoir',
			4: 'Albrecht Durer',
			5: 'Paul Gauguin',
			6: 'Francisco Goya',
			7: 'Rembrandt',
			8: 'Alfred Sisley',
			9: 'Titian',
			10: 'Marc Chagall'
		  }

    def get_model_prediction(input_url):
        print("Predicting")
        train_input_shape = (224, 224)
        web_image = imageio.imread(input_url)
        web_image = Image.fromarray(web_image)
        web_image = web_image.resize(train_input_shape[0:2])
        web_image = tf.keras.preprocessing.image.img_to_array(web_image)
        web_image /= 255.
        web_image = np.expand_dims(web_image, axis=0)

        # Test model on input data.
        input_shape = input_details[0]['shape']

        # Feed the image data
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_tensor= tf.convert_to_tensor(input_data, np.float32)
        interpreter.set_tensor(input_details[0]['index'], web_image)

        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction_probability = np.amax(prediction) * 100
        prediction_probability = "{0:.2f}%".format(prediction_probability)
        prediction_idx = np.argmax(prediction)

        output_data = {'artist': artists[prediction_idx], 'probability': prediction_probability}

        return output_data
    prediction = get_model_prediction(input_url)
    print(prediction)
    return prediction
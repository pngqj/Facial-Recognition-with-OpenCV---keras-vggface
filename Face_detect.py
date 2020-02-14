import tensorflow as tf
from PIL import Image
import numpy as np 
from numpy import expand_dims
from keras_vggface.utils import preprocess_input 
import pickle 


class Face_detect():
	def __init__(self):
		self.model = tf.keras.models.load_model('train_model/saved_model.hdf')
		with open('train_model/labels.pickle', 'rb') as handle:
			# a pickle file containing a dictionory of id as key, and names as values
			# generated from train_model/train.py
			self.labels = pickle.load(handle)
			print(self.labels)

	def extract_face(self, face):
		# modify and resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize((224, 224))
		face_array = np.asarray(image)
		# convert face into samples
		pixels = face_array.astype('float32')
		samples = expand_dims(pixels, axis=0)
		# prepare the face for the model, e.g. center pixels
		samples = preprocess_input(samples, version=2)
		return samples
		
	def predict_face(self, face):
		predicted = self.model.predict(self.extract_face(face))
		n = np.argmax(predicted, axis=1)[0]
		score = predicted[0][n]
		print(self.labels[n], score)
		return self.labels[n]
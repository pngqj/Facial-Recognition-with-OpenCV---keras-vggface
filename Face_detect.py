import tensorflow as tf
from PIL import Image
import numpy as np 
from numpy import expand_dims
from keras_vggface.utils import preprocess_input 
import pickle 
import cv2

class FacialRecognition():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('train_model/haarcascade_frontalface_alt2.xml')
        self.model = tf.keras.models.load_model('train_model/saved_model.hdf')
        with open('train_model/labels.pickle', 'rb') as handle:
            # a pickle file containing a dictionory of id as key, and names as values
            # generated from train_model/train.py
            self.labels = pickle.load(handle)

    def predict_face(self, data):
        extracted_face = self.extract_face_from_img(data)
        try:
            predicted = self.model.predict(self.resize_face(extracted_face))
            n = np.argmax(predicted, axis=1)[0]
            score = predicted[0][n]
            return [self.labels[n], score]
        except:
            return [None, None]

    def extract_face_from_img(self, data):
        try:
          img = cv2.imdecode(np.fromstring(data.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        except:
          img = data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)

        if faces != ():
            (x, y, w, h) = faces[0]
            roi_color = img[y:y+h, x:x+w]
            return roi_color
        else:
            return None
    
    def resize_face(self, face):
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
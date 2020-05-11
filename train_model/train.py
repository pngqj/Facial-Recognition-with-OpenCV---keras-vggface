# pip install git+https://github.com/rcmalli/keras-vggface.git

from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from keras_vggface.vggface import VGGFace
from numpy import expand_dims
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from PIL import ImageEnhance
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Model
from keras.layers import Flatten
from keras import optimizers 

def extract_face(face, rotate_angle = 0, enhance_factor=1):
  # modify and resize pixels to the model size
  image = Image.fromarray(face)
  image = image.rotate(rotate_angle)
  enhancer = ImageEnhance.Contrast(image)
  image = enhancer.enhance(enhance_factor)
  image = image.resize((224, 224))
  face_array = np.asarray(image)
  # convert face into samples
  pixels = face_array.astype('float32')
  samples = expand_dims(pixels, axis=0)
  # prepare the face for the model, e.g. center pixels
  samples = preprocess_input(samples, version=2)
  return samples

import cv2

def detect_face(path):
  img = cv2.imread(path)
  try:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  except:
    return None
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # remember to add xml to path
  faces = face_cascade.detectMultiScale(gray)
  if faces != ():
    (x, y, w, h) = faces[0]
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_color = img[y:y+h, x:x+w]
    return roi_color


sample_path = "morgan freeman/1.jpg"
sample_face_array = detect_face(sample_path)
sample_face_array = extract_face(sample_face_array, rotate_angle=0,enhance_factor=1)
sample_face_array.shape

# format: ('person's name': 'path to image file')

dataset = [
            ('trump', "trump/1.jpg"),
            ('trump', "trump/2.jpg"),
            ('trump', "trump/3.jpg"),
            ('morgan freeman', "morgan freeman/1.jpg"),
            ('morgan freeman', "morgan freeman/2.jpg"),
            ('morgan freeman', "morgan freeman/3.jpg"),
            ('nelson mandela', "nelson mandela/1.jpg"),
            ('nelson mandela', "nelson mandela/2.jpg"),
            ('nelson mandela', "nelson mandela/3.jpg"),           
            ('ignore', "ignore/1.jpg"),
            ('ignore', "ignore/2.jpg"),
            ('ignore', "ignore/3.jpg"),
            ('ignore', "ignore/4.jpg"),
]


num_class = len(set([x[0] for x in dataset]))
print("Number of class:", num_class)



faces = []
face_y = []

for (y_,x_) in dataset:
  detect = detect_face(x_)

  if detect is not None:
    faces.append(detect)
    face_y.append(y_)
  else:
    print(x_)

x = []
y = []

enhance_list = [0.5 , 0.75, 1.0 , 1.25, 1.5 ]
for enhance_factor in enhance_list:
  print(enhance_factor)
  for angle in range(-30, 30, 5):
    for f, y_ in zip(faces,face_y):
      data = extract_face(f, rotate_angle=angle, enhance_factor=enhance_factor)
      x.append(data)
      y.append(y_)


x = np.array(x).reshape(-1,224,224,3)
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y), num_classes=num_class)


with open('labels.pickle', 'wb') as handle:
  labels = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
  print(labels)
  pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


model = VGGFace(include_top=False, input_shape=(224, 224, 3), model='resnet50')

for layer in model.layers:
	layer.trainable = False

last_layer = model.get_layer('avg_pool').output
x_ = Flatten(name='flatten')(last_layer)
x_ = Dense(512, activation='relu', name='fc6')(x_)
x_ = Dense(512, activation='relu', name='fc7')(x_)
out = Dense(num_class, activation='sigmoid', name='fc8')(x_)
model = Model(model.input, out)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping

save_best = ModelCheckpoint('saved_model.hdf', save_best_only=True, monitor='val_loss', mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')

result = model.fit(x_train, y_train, batch_size = 20, epochs=100, 
                    callbacks=[earlyStopping, save_best],
                    validation_data=(x_test,y_test), verbose=1)

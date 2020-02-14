from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions
from PIL import ImageEnhance
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical 
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping



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


def detect_face(path):
  print(path)
  img = cv2.imread(path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
  faces = face_cascade.detectMultiScale(gray)
  if faces != ():
    (x, y, w, h) = faces[0]
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_color = img[y:y+h, x:x+w]
    return roi_color

# format: ('person's name': 'path to image file')

dataset = [
           ('Donald Trump', "trump/1.jpg"),
           ('Morgan Freeman', "morgan freeman/1.jpg"),
           ('Morgan Freeman', "morgan freeman/5.jpg"),
           ('Donald Trump', "trump/2.jpg"),
           ('Donald Trump', "trump/3.jpg"),
           ('Morgan Freeman', "morgan freeman/2.jpg"),
           ('Donald Trump', "trump/4.jpg"),
           ('Morgan Freeman', "morgan freeman/4.jpg"),
           ('Donald Trump', "trump/5.jpg"),
           ('Donald Trump', "trump/6.jpg"),
           ('Morgan Freeman', "morgan freeman/3.jpg"),
           ('Morgan Freeman', "morgan freeman/6.jpg"),
           ('Nelson Mandela', "nelson mandela/1.jpg"),
           ('Nelson Mandela', "nelson mandela/2.jpg"),
           ('Nelson Mandela', "nelson mandela/3.jpg"),
           ('Nelson Mandela', "nelson mandela/4.jpg"),
           ('QJ', "QJ/1.jpg"),
           ('QJ', "QJ/2.jpg"),
           ('QJ', "QJ/3.jpg"),
           ('QJ', "QJ/4.jpg"),
]

num_class = len(set([x[0] for x in dataset]))
print("Number of class:", num_class)

y = []
x = []

faces = []
for (y_,x_) in dataset:
  detect = detect_face(x_)
  faces.append(detect)

# probably a little overkill for training.
enhance_list = [0.5 , 0.75, 1.0 , 1.25, 1.5 ]
for enhance_factor in enhance_list:
  print(enhance_factor)
  for angle in range(-30, 30, 5):
    for f, (y_,x_) in zip(faces, dataset):
      y.append(y_)
      data = extract_face(f, rotate_angle=angle, enhance_factor=enhance_factor)
      x.append(data)


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
out = Dense(num_class, activation='softmax', name='fc8')(x_)
model = Model(model.input, out)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

save_best = ModelCheckpoint('saved_model.hdf', save_best_only=True, monitor='val_loss', mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

model.fit(x_train, y_train, batch_size = 20, epochs=100, 
                callbacks=[earlyStopping, save_best],
                validation_data=(x_test,y_test), verbose=1)
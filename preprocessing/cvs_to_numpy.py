import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join

import  sys
sys.path.insert(0, "../constants")
try:
  from constants import SIZE_FACE,EMOTIONS,SAVE_DIRECTORY,SAVE_MODEL_FILENAME,CASC_PATH,SAVE_TRAINING_IMAGES_FILENAME,\
      SAVE_TRAINING_LABELS_FILENAME,SAVE_VALIDATION_IMAGES_FILENAME,SAVE_VALIDATION_LABELS_FILENAME,TRAINING_DATA_PERCENTAGE,TOTAL_DATASET_COUNT,DATASET_DIRECTORY
except ImportError:
  print('No Import')


cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  gray_border = np.zeros((150, 150), np.uint8)
  gray_border[:,:] = 200
  gray_border[(int((150 / 2)) - int((SIZE_FACE / 2))):int(((150 / 2) + (SIZE_FACE / 2))), int(((150 / 2) - (SIZE_FACE / 2))):int(((150 / 2) + (SIZE_FACE / 2)))] = image
  image = gray_border

  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is we don't found an image
  if not len(faces) > 0:
    #print "No hay caras"
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resize image to network size

  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("-------------Problem encountered while resizing the image-----------------")
    return None
  return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

def flip_image(image):
    return cv2.flip(image, 1)

def data_to_image(data):
    data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy() 
    data_image = format_image(data_image)
    return data_image

FILE_PATH = '../dataset/fer2013.csv'
data = pd.read_csv(FILE_PATH)

labels_training = []
images_training = []
labels_validation = []
images_validation = []
index = 1
final_data=np.shape(data)
total = data.shape[0]
count=1
error_data=0
validation_data_count=0;
training_data_count=0
print('----------Starting processing data--------------')
for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    if image is not None:
        if not count > int (TRAINING_DATA_PERCENTAGE * TOTAL_DATASET_COUNT):
            labels_training.append(emotion)
            images_training.append(image)
            training_data_count=training_data_count+1
        else:
            labels_validation.append(emotion)
            images_validation.append(image)
            validation_data_count=validation_data_count+1
        count=count+1
    else:
        data=data.drop(index)
        error_data=error_data+1
    index += 1
print('----------Data processed and segregated in validation and training sets--------------')
print()
print("Total Validation Images Count: "+str(validation_data_count))
print("Total Training Images Count: "+str(training_data_count))
print("Total Error Data: "+str(error_data))
np.save(join(DATASET_DIRECTORY,SAVE_TRAINING_IMAGES_FILENAME), images_training)
np.save(join(DATASET_DIRECTORY,SAVE_TRAINING_LABELS_FILENAME), labels_training)
np.save(join(DATASET_DIRECTORY,SAVE_VALIDATION_IMAGES_FILENAME), images_validation)
np.save(join(DATASET_DIRECTORY,SAVE_VALIDATION_LABELS_FILENAME), labels_validation)
np.save(join(DATASET_DIRECTORY,'fer2013.npy'),data)
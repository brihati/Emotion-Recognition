from __future__ import division, absolute_import

import sys
from os.path import isfile, join
import tflearn
from constants.constants import *
from preprocessing import DatasetLoader
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization


class emotion_cnn:
  def __init__(self):
    self.dataset = DatasetLoader()

  def build_network(self):
    print('---------------------Building CNN---------------------')

    img_preProcess=ImagePreprocessing()
    img_preProcess.add_featurewise_zero_center()
    img_preProcess.add_featurewise_stdnorm()

    #img_aug=ImageAugmentation()
    #img_aug.add_random_flip_leftright()
    #img_aug.add_random_rotation(max_angle=25)
    #img_aug.add_random_blur(sigma_max=3)

    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1],
                              data_preprocessing = img_preProcess)
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = local_response_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 128, 4, activation = 'relu')
    self.network = dropout(self.network, 0.3)
    self.network = fully_connected(self.network, 3072, activation = 'relu')
    self.network = fully_connected(self.network, len(EMOTIONS), activation ='softmax')
    self.network = regression(self.network,
      optimizer = 'adam',
      loss = 'categorical_crossentropy',learning_rate=0.001)
    self.model = tflearn.DNN(
      self.network,
      tensorboard_dir=DATASET_DIRECTORY,
      checkpoint_path =DATASET_DIRECTORY + '/emotion_recognition',
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )
    self.load_model()
    print('-----------------------Model Loaded----------------------')

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('----------------Dataset found and loaded-----------------')

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('--------------------Training network----------------------')
    print('Images validation:'+str(len(self.dataset._images_validation)))
    print('Labels Validation:'+str(len(self.dataset._labels_validation)))
    print ('Images training'+str(len (self.dataset._images)))
    print ('Labels training'+str(len (self.dataset._labels)))
    self.model.fit(
      self.dataset._images, self.dataset._labels,
      validation_set = (self.dataset._images_validation, self.dataset._labels_validation),
      n_epoch = 100,
      batch_size = 50,
      shuffle = True,
      show_metric = True,
      snapshot_step = 200,
      snapshot_epoch = True,
      run_id = 'emotion_recognition'
    )

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(DATASET_DIRECTORY, SAVE_MODEL_FILENAME))
    print('---------------Model trained and saved at ' + SAVE_MODEL_FILENAME + '-------------------')

  def load_model(self):
    if isfile(join(DATASET_DIRECTORY, SAVE_MODEL_FILENAME)):
      self.model.load(join(DATASET_DIRECTORY, SAVE_MODEL_FILENAME))
      print('---------------------Model loaded from ' + SAVE_MODEL_FILENAME + '------------------------')

network=emotion_cnn()
network.start_training ()
print ('---------------------Network Trained-----------------------------')
network.save_model ()

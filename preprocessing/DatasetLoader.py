from os.path import join
from constants.constants import *
import numpy as np


class DatasetLoader(object):

  def __init__(self):
    pass

  def load_from_save(self):
    self._images      = np.load(join(DATASET_DIRECTORY, SAVE_TRAINING_IMAGES_FILENAME))
    self._labels      = np.load(join(DATASET_DIRECTORY, SAVE_TRAINING_LABELS_FILENAME))
    self._images      = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels      = self._labels.reshape([-1, len(EMOTIONS)])
    self._images_validation = np.load (join (DATASET_DIRECTORY, SAVE_VALIDATION_IMAGES_FILENAME))
    self._labels_validation = np.load (join (DATASET_DIRECTORY, SAVE_VALIDATION_LABELS_FILENAME))
    self._labels_validation = self._labels_validation.reshape ([-1, len (EMOTIONS)])
    self._images_validation = self._images_validation.reshape ([-1, SIZE_FACE, SIZE_FACE, 1])

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_validation(self):
    return self._images_validation

  @property
  def labels_validation(self):
    return self._labels_validation

  @property
  def num_examples(self):
    return self._num_examples
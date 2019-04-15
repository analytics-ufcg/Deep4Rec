'''A dataset abstraction.'''

from abc import ABC
from abc import abstractmethod
import logging

import deep4rec.datasets.utils as ds_utils


class Dataset(ABC):
  '''Dataset interface.'''

  url = None

  def __init__(self, dataset_name, output_dir, verbose=False, *args, **kwargs):
    self.dataset_name = dataset_name
    self.output_dir = output_dir
    self.verbose = verbose
 
  def download(self):
    if self.verbose:
      logging.info('Downloading {} at {}'.format(self.dataset_name, self.output_dir))
    ds_utils.download(self.url, self.output_dir)

  @abstractmethod
  def preprocess(self):
    raise NotImplementedError

  @abstractmethod
  def check_downloaded(self):
    '''Checks if the downloaded files already exist in `path`.'''
    raise NotImplementedError

  @abstractmethod
  def check_preprocessed(self):
    '''Checks if the expected preprocessed files exist in `path`.'''
    raise NotImplementedError

  def maybe_download(self):
    if not self.check_downloaded():
      self.download()

  def maybe_preprocess(self):
    if not self.check_preprocessed():
      self.preprocess()
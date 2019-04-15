"""Utilitiy functions for datasets."""

import os
import urllib.request
import zipfile


DEFAULT_OUTPUT_DIR = '/tmp/tfrec_data/'


def parent_path(path):
  '''Finds parent path.

  Args:
    path: A string indicating a path.
  
  Returns:
    A string indicating the parent path of `path`.
  '''
  return os.path.dirname(path)


def base_name(path):
  '''Identifies base name of `path`.

  Args:
    path: A string indicating a path.

  Returns:
    A string indicating the base name of `path`.
  '''
  return os.path.basename(path)


def unzip(file_path, output_path=None):
  '''Unzips a file.

  Args:
    file_path: A string indicating a zip file path.
    output_path: A string indicating the parent file path. Optional, if
    None then the output path will be the parent of `file_path`. 
  '''
  if not output_path:
    output_path = parent_path(file_path)

  zip_obj = zipfile.ZipFile(file_path)
  maybe_mkdir(output_path) 
  zip_obj.extractall(output_path)


def maybe_unzip(file_path, output_path=None):
  '''If file path is a zip file than unzips.'''
  for file_ext in ['.tar.gz', '.bz2', '.zip']:
    if file_path.endswith(file_ext):
      unzip(file_path, output_path=output_path)


def maybe_mkdir(path, is_file=False):
  dir_path = parent_path(path) if is_file else path
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def download(url, output_dir):
  '''Download file from `url` and store at `output_path`.'''
  maybe_mkdir(output_dir)
  url_basename = base_name(url)
  urllib.request.urlretrieve(url,
                             filename=os.path.join(output_dir, url_basename))

"""Utilitiy functions."""
import gzip
import numpy as np
import os
import shutil
import tensorflow as tf
import urllib.request
import zipfile


DEFAULT_OUTPUT_DIR = "/tmp/deep4rec_data/"

# SECTION: Path related funtions


def parent_path(path):
    """Finds parent path.

    Args:
      path: A string indicating a path.

    Returns:
      A string indicating the parent path of `path`.
    """
    return os.path.dirname(path)


def base_name(path):
    """Identifies base name of `path`.

    Args:
      path: A string indicating a path.

    Returns:
      A string indicating the base name of `path`.
    """
    return os.path.basename(path)


def unzip(file_path, output_path=None):
    """Unzips a file.

    Args:
      file_path: A string indicating a zip file path.
      output_path: A string indicating the parent file path. Optional, if
      None then the output path will be the parent of `file_path`.
    """
    if not output_path:
        output_path = parent_path(file_path)

    zip_obj = zipfile.ZipFile(file_path)
    maybe_mkdir(output_path)
    zip_obj.extractall(output_path)


def ungz(file_path, output_path=None):
    """Uncompress .gz files.

    Args:
      file_path: A string indicating a compressed file path.
      output_path: A string indicating the parent file path. Optional, if
      None then the output path will be the parent of `file_path`.
    """
    if not output_path:
        output_path = parent_path(file_path)

    maybe_mkdir(output_path)

    file_name = file_path.replace(".tar.gz", "")
    file_name = file_name.replace(".gz", "")

    with gzip.open(file_path, "rb") as f_in:
        with open(file_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


uncompress_functions = {"zip": unzip, "gz": ungz, ".tar.gz": ungz}


def maybe_uncompress(file_path, output_path=None):
    """If file path is a compressed file than uncompress it."""
    for file_ext in uncompress_functions:
        if file_path.endswith(file_ext):
            uncompress_functions[file_ext](file_path, output_path=output_path)


def maybe_mkdir(path, is_file=False):
    dir_path = parent_path(path) if is_file else path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def download(url, output_dir):
    """Download file from `url` and store at `output_path`."""
    maybe_mkdir(output_dir)
    url_basename = base_name(url)
    urllib.request.urlretrieve(url, filename=os.path.join(output_dir, url_basename))


# SECTION: Name to function


def names_to_fn(names, mapper_fn):
    return [name_to_fn(name, mapper_fn) for name in names]


def name_to_fn(fn, mapper_fn):
    """If needed maps a name to a function based on `mapper_fn`."""
    if isinstance(fn, str):
        fn = mapper_fn(fn)
    return fn


# SECTION: logits to pred/class


def logits_to_class(pred, th=0.5):
    if isinstance(pred[0], list) and len(pred[0]) > 1:
        return tf.argmax(pred).numpy()
    else:
        return np.array(pred) > th


def softmax(z):
    z = np.array(z)
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def sigmoid(z):
    return 1 / (1 + np.exp(-np.array(z)))


def logits_to_prob(pred):
    if isinstance(pred[0], list) and len(pred[0]) > 1:
        return softmax(pred)
    else:
        return sigmoid(pred)

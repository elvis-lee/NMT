import errno
import os
import tensorflow as tf
import urllib



def maybe_download(url, local_dir, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    mkdir_p(local_dir)
    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(local_dir, local_filename)
    if not os.path.exists(local_filepath):
        print("Downloading %s..." % local_filename)
        local_filename, _ = urllib.request.urlretrieve(url,
                                                       local_filepath)
        print("Finished!")
    statinfo = os.stat(local_filepath)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', local_filepath)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename


def mkdir_p(path):
    """From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


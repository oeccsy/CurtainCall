import tensorflow

from tensorflow.python.client import device_lib

print(tensorflow.__version__)

print(device_lib.list_local_devices())

tensorflow.__version__
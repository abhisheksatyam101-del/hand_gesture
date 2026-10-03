#!/bin/bash
source /home/abhis/venv_wsl/bin/activate
python3 -c "import tensorflow as tf; print('GPUs detected: ', tf.config.list_physical_devices('GPU'))"

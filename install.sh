#!/bin/bash

rm -rf /mnt/anindya/trainer
virtualenv --system-site-packages /mnt/anindya/trainer
source /mnt/anindya/trainer/bin/activate
pip install --upgrade pip
pip install -I -r /mnt/anindya/requirements.txt

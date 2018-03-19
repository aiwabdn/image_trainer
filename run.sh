#!/bin/bash

rm -rf tmp/*
source /mnt/anindya/trainer/bin/activate
nohup python -u train.py &>> log &

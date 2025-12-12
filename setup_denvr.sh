#!/usr/bin/bash

# this scripts setups the mounts and unpacks the data on denvr

# set up fast local storage
sudo mkfs -t ext4 /dev/vdc

# create tmps storage for unpacking data
mkdir -p /home/ubuntu/scratch

# mount the direct attached to the new storage
sudo mount /dev/vdc /home/ubuntu/scratch

# change the ownership to ubuntu user
sudo chown ubuntu:ubuntu /home/ubuntu/scratch

# create data directory
mkdir -p /home/ubuntu/scratch/cellanome/{SMALL_TRAINING_DATA2,TRAINING_DATA}
echo "Data directories created"

# unpack data from /personal to local scratch
echo "Unpacking small-data.tar.xz data..."
time tar -xvf /mnt/personal/small-data.tar.xz -C /home/ubuntu/scratch/cellanome/
echo "Unpacking small-data.tar.xz complete."

# unpack all data from /personal to local scratch
echo "Unpacking train.tar.xz data..."
time tar -xvf /mnt/personal/train.tar.xz -C /home/ubuntu/scratch/cellanome/TRAINING_DATA
echo "Unpacking train.tar.xz complete."
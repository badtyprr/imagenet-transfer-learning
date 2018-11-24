#!/bin/sh

if [ $# -ne 2 ]; then
  exit 1
fi

if [ ! -f $1 ]; then
  echo "download $2"
  wget $2 -O $1 -T 1 -t 5 -nc -b -a wget.log
  # Downloads ImageNet in about 4 days
  sleep 0.25
fi

if [ -f $1 ]; then
  echo "file $1 exists"
fi


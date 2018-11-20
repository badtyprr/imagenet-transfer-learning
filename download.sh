#!/bin/sh

if [ $# -ne 2 ]; then
  exit 1
fi

wget $2 -O $1 -T 1 -t 5 -nc -b -a wget.log
# Downloads ImageNet in about 4 days
sleep 0.25

#!/bin/sh

if [ $1 == "CUDA" ]; then
  NB=$(nvidia-smi -L | wc -l)
  if [ $NB -gt $2 ]; then
    exit 0
  fi
fi

if [ $1 == "HIP" ]; then
  NB=$(rocm-smi --showuniqueid | grep 'Unique ID:' | wc -l)
  if [ $NB -gt $2 ]; then
    exit 0
  fi
fi

exit 1

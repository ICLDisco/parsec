#!/bin/sh

echo HTOD - GPU 0
cat /dev/null | ./bandwidthTest --device=0 --memory=pinned --mode=range --start=65536 --end=1048576 --increment=32768 --htod > perfs-gpu0-htod.dat
cat /dev/null | ./bandwidthTest --device=0 --memory=pinned --mode=range --start=1048576 --end=12582912 --increment=262144 --htod >> perfs-gpu0-htod.dat

echo DTOH - GPU 0
cat /dev/null | ./bandwidthTest --device=0 --memory=pinned --mode=range --start=65536 --end=1048576 --increment=32768 --dtoh > perfs-gpu0-dtoh.dat
cat /dev/null | ./bandwidthTest --device=0 --memory=pinned --mode=range --start=1048576 --end=12582912 --increment=262144 --dtoh >> perfs-gpu0-dtoh.dat

echo HTOD - GPU 1
cat /dev/null | ./bandwidthTest --device=1 --memory=pinned --mode=range --start=65536 --end=1048576 --increment=32768 --htod > perfs-gpu1-htod.dat
cat /dev/null | ./bandwidthTest --device=1 --memory=pinned --mode=range --start=1048576 --end=12582912 --increment=262144 --htod >> perfs-gpu1-htod.dat

echo DTOH - GPU 1
cat /dev/null | ./bandwidthTest --device=1 --memory=pinned --mode=range --start=65536 --end=1048576 --increment=32768 --dtoh > perfs-gpu1-dtoh.dat
cat /dev/null | ./bandwidthTest --device=1 --memory=pinned --mode=range --start=1048576 --end=12582912 --increment=262144 --dtoh >> perfs-gpu1-dtoh.dat

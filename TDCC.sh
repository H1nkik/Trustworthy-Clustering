#!/bin/sh
#JSUB -q gpu
#JSUB -n 4
#JSUB -gpgpu 1
#JSUB -e errorTDCC-STL.txt
#JSUB -o outputTDCC-STL.txt
#JSUB -J TDCCGPU
./data/users/zhouk/software/anaconda3/bin/python3
./TDCC.py

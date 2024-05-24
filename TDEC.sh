#!/bin/sh
#JSUB -q gpu
#JSUB -n 4
#JSUB -gpgpu 1
#JSUB -e errorTDEC-STL.txt
#JSUB -o outputTDEC-STL.txt
#JSUB -J TDECGPU
./data/users/zhouk/software/anaconda3/bin/python3
./TDEC.py
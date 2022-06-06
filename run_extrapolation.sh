#! /bin/bash

x=${1:-1};

while [ $x -gt 0 ]; do
   echo "starting run";
   python run_extrapolation.py --seed $x;
   x=$(($x-1));
done


python analyze.py;
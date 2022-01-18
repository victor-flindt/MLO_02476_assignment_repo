#!/bin/bash
# A sample Bash script, by Ryan
x=1
while [ $x -le 5 ]
do
  python3 helloworld.py
  x=$(( $x + 1 ))
done

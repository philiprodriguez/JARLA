#! /bin/bash

echo "Running JARLA..."
cd /home/pi/Desktop/JARLA
source environment/bin/activate
python3 jarla.py
echo "JARLA is dead!"
sleep 99999

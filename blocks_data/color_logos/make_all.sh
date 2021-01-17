#!/bin/bash

for file in *.png; do
        echo ${file} 
        filename="${file%.*}"
        python make_mtl.py ${filename}
        python make_cube.py ${filename}
done

#!/bin/bash
#brew install ffmpeg
mkdir Resized
for f in *.png; do ffmpeg -i "$f" -s 128x128 "./Resized/${f/%.png/.png}"; done

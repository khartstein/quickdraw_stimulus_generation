#!/bin/bash
# make 6s videos by looping 2s videos 3 times for unambiguous
# stimuli in the sketch-morph experiment using ffmpeg
#
# Notes:
# Run in choose-stimuli conda environment from the directory that
# contains this file. Obviously, you also need to create the 2s videos first.

pwd=$(pwd)
stimList=$(ls ${pwd}/stimuli/unambiguous_videos_resized/)

for var in $stimList
  do
    ffmpeg -stream_loop 2 \
      -i ${pwd}/unambiguous_videos_resized/${var} \
      -c copy ${pwd}/unambiguous_videos_resized/${var%.*}_6s.mov
    done

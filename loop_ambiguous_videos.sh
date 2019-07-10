#!/bin/bash
# make 6s videos by looping 2s videos 3 times for unambiguous
# stimuli in the sketch-morph experiment using ffmpeg
#
# Notes:
# run this script in the choose-stimuli conda environment from the
# directory it is in

pwd=$(pwd)
stimList=$(ls ${pwd}/stimuli/ambiguous_videos_resized/short/)

for var in $stimList
  do
    ffmpeg -stream_loop 2 \
      -i ${pwd}/stimuli/ambiguous_videos_resized/short/${var} \
      -c copy ${pwd}/stimuli/ambiguous_videos_resized/looped/${var%.*}_6s.mov
    done

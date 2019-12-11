#!/bin/bash

for zips in */*-points.zip; do
    output=`echo $zips | awk '{print substr($0, 1, length($0)-length(".zip"))}'`
    mkdir -p $output
    unzip -o "$zips" -d "$output"
done

python extract_patches.py

exit 0

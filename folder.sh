#!/bin/bash


# Used to find the required files in each folder
mkdir 17_06 20_06
radar_folders=(radar_44 radar_51 radar_52 radar_53 radar_111 radar_112 radar_113 radar_114)
cd 15_06
for folder in "${radar_folders[@]}"; do
    # Create the radar folder if it doesn't exist
    mkdir -p "$folder"

    # Create part folders inside the radar folder
    for i in {1..7}; do
        mkdir -p "$folder/part_$i"
    done
done
cd ..

cd 16_06
for folder in "${radar_folders[@]}"; do
    # Create the radar folder if it doesn't exist
    mkdir -p "$folder"

    # Create part folders inside the radar folder
    for i in {1..7}; do
        mkdir -p "$folder/part_$i"
    done
done
cd ..

cd 17_06
for folder in "${radar_folders[@]}"; do
    # Create the radar folder if it doesn't exist
    mkdir -p "$folder"

    # Create part folders inside the radar folder
    for i in {1..7}; do
        mkdir -p "$folder/part_$i"
    done
done
cd ..

cd 20_06
for folder in "${radar_folders[@]}"; do
    # Create the radar folder if it doesn't exist
    mkdir -p "$folder"

    # Create part folders inside the radar folder
    for i in {1..7}; do
        mkdir -p "$folder/part_$i"
    done
done
cd ..

#!/bin/bash

# Number of runs
num_runs=30

# Initialize sum of FPS
sum_fps=0

# Loop to run the command multiple times
for ((i=1; i<=num_runs; i++))
do
    echo "Run #$i"
    # Run the command and capture the output
    output=$(./resnet50_pt_fps resnet50_DYB.xmodel ../DYB-linearHead/test)
    
    # Extract the Average FPS value using grep and awk
    fps=$(echo "$output" | grep "Average FPS:" | awk '{print $3}')
    
    # Print the Average FPS for the current run
    echo "Run #$i finished, average FPS = $fps"
    
    # Add the FPS value to the sum
    sum_fps=$(echo "$sum_fps + $fps" | bc)
done

# Calculate the mean FPS
mean_fps=$(echo "scale=4; $sum_fps / $num_runs" | bc)

# Print the mean FPS
echo "Mean Average FPS: $mean_fps"

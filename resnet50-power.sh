#!/bin/bash

# Number of runs
num_runs=30

# Initialize sum of FPS
sum_fps=0

# Start monitor_power.sh script
./monitor_power.sh &
MONITOR_PID=$!

# Loop to run the command multiple times
for ((i=1; i<=num_runs; i++))
do
    echo "Run #$i"
    # Run the command and capture the output
    #output=$(./resnet50_pt_fps resnet50_int8.xmodel ../DYB-linearHead/test)
    #output=$(./resnet50_pt_fps resnet50_iterative.xmodel ../DYB-linearHead/test)
    #output=$(./resnet50_pt_fps resnet50_one_step.xmodel ../DYB-linearHead/test)
    output=$(./resnet50_pt_fps resnet50_ofa.xmodel ../DYB-linearHead/test)
    #output=$(./resnet50_pt_fps resnet50_fastNAS.xmodel ../DYB-linearHead/test)

    # Extract the Average FPS value using grep and awk
    fps=$(echo "$output" | grep "Average FPS:" | awk '{print $3}')
    
    # Print the Average FPS for the current run
    echo "Run #$i finished, average FPS = $fps"
    
    # Add the FPS value to the sum
    sum_fps=$(echo "$sum_fps + $fps" | bc)
done

# End monitor_power.sh script
kill -TERM $MONITOR_PID
wait $MONITOR_PID

# Calculate the mean FPS
mean_fps=$(echo "scale=4; $sum_fps / $num_runs" | bc)

# Print the mean FPS
echo "Mean Average FPS: $mean_fps"

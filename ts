#!/usr/bin/env python
import sys
import datetime

assert len(sys.argv) > 1
# Open the file with the timestamps
with open(sys.argv[1]) as file:
    # Read each line of the file
    lines = file.readlines()
# Convert each line to a datetime object
times = []
valid_lines = []
for line in lines:
    try:
        times.append(datetime.datetime.strptime(line.strip().split()[0], "%H:%M:%S"))
        valid_lines.append(line)
    except:
        continue
ops = [" ".join(line.strip().split()[1:]) for line in valid_lines]
# Print the time difference between each timestamp
for i in range(1, len(times)):
    print(times[i] - times[i-1], ops[i-1])

#!/usr/bin/env python
import sys
import datetime

base = "00:00:00:000"
    

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
        times.append(datetime.datetime.strptime(line.strip().split()[1], "%H:%M:%S:%f"))
        valid_lines.append(line)
    except:
        continue
ops = [" ".join(line.strip().split()[2:]) for line in valid_lines]
# Print the time difference between each timestamp
nulldt = datetime.datetime(1900, 1, 1, 0, 0, 0) 
for i in range(1, len(times)):
    time = nulldt + (times[i] - times[i-1])
    time = (time).strftime("%H:%M:%S.%f")
    print(time[:10], ops[i-1])

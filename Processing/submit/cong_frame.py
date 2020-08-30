import csv
import numpy as np
import math


result = 'submit_processed.txt'
file_in = 'submit.txt'
print(file_in)
frame_plus = 5
with open(file_in, "r") as inf:
    reader = csv.reader(inf, delimiter=' ')
    with open(result, "w") as out:   
        for row in reader:
            if int(row[1]) - frame_plus < 1:
                continue
            else:
                out.write("{} {} {} {}\n".format(row[0],(int(row[1]) - frame_plus),row[2],row[3]))
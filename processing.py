import csv
import numpy as np
import math
import glob

"""
result = 'submit/cam_11_submit.txt'
file_in = 'cam_11_result.txt'
print(file_in)
frame_plus = 0
with open(file_in, "r") as inf:
    reader = csv.reader(inf, delimiter=' ')
    with open(result, "w") as out:   
        for row in reader:
            if int(row[3]) != 0:
                out.write("{} {} {} {}\n".format(row[0],int(row[1])+frame_plus,row[2],row[3]))
"""



submit = "data/submission_output/submission.txt"
out = open(submit,"w")
frame_plus = 0
for file in glob.glob("submit/*.txt"):
    print(file)
    #name = os.path.split(file)[-1]
    #path_in = "{}_result.txt".format(name)
    with open(file, "r") as inf:
        reader = csv.reader(inf, delimiter=' ')
        for row in reader:
            if int(row[3]) != 0:
                out.write("{} {} {} {}\n".format(row[0],int(row[1])+frame_plus,row[2],row[3]))
out.close()

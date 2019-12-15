import numpy as np
import re
import os





read = open("prey_nn.svg","r")
write = open("prey_nn_tmp.svg","w")

while True:
    c = read.read(1)
    if not c:
        break
    write.write(c)
    if c == ">":
        write.write("\n")

read.close()
write.close()


read = open("pred_nn.svg","r")
write = open("pred_nn_tmp.svg","w")

while True:
    c = read.read(1)
    if not c:
        break
    write.write(c)
    if c == ">":
        write.write("\n")

read.close()
write.close()





nn = np.load("best_prey.npy","r")
read = open("prey_nn_tmp.svg","r")
write = open("prey_nn_weight.svg","w")

max = nn.max()
min = nn.min()

i=0
j=0
linear = 1
while True:
    line = read.readline()
    if not line:
        break
    if not re.search(r'^.*stroke-opacity: ([0-9|.]*);.*$', line):
        write.write(line)
    else:
        opacity = np.abs(np.tanh(nn[i]/4))
        color = int((1-opacity)*255)
        new_line_tmp = re.sub(r'stroke-opacity: [0-1]\.[0-9]*;', "stroke-opacity: {};".format(opacity), line)
        if nn[i] <= 0:
            new_line = re.sub(r'stroke: rgb\(.*\);', "stroke: rgb(255,{},{});".format(color,color), new_line_tmp)
        else:
            new_line = re.sub(r'stroke: rgb\(.*\);', "stroke: rgb({},{},255);".format(color,color), new_line_tmp)

        write.write(new_line)
        i+=1
        j+=1
        if linear == 1 and j == 288: #24*12
            i+=12
            j=0
            linear = 2

        if linear == 2 and j == 48: #12*4
            i+=4
            j=0

read.close()
write.close()


nn = np.load("best_pred.npy","r")
read = open("pred_nn_tmp.svg","r")
write = open("pred_nn_weight.svg","w")

max = nn.max()
min = nn.min()

i=0
j=0
linear = 1
while True:
    line = read.readline()
    if not line:
        break
    if not re.search(r'^.*stroke-opacity: ([0-9|.]*);.*$', line):
        write.write(line)
    else:
        opacity = np.abs(np.tanh(nn[i]/5))
        color = int((1-opacity)*255)
        new_line_tmp = re.sub(r'stroke-opacity: [0-1]\.[0-9]*;', "stroke-opacity: {};".format(opacity), line)
        if nn[i] <= 0:
            new_line = re.sub(r'stroke: rgb\(.*\);', "stroke: rgb(255,{},{});".format(color,color), new_line_tmp)
        else:
            new_line = re.sub(r'stroke: rgb\(.*\);', "stroke: rgb({},{},255);".format(color,color), new_line_tmp)

        write.write(new_line)
        i+=1
        j+=1
        if linear == 1 and j == 144: #12*12
            i+=12
            j=0
            linear = 2

        if linear == 2 and j == 48: #12*4
            i+=4
            j=0

read.close()
write.close()




os.remove("prey_nn_tmp.svg")
os.remove("pred_nn_tmp.svg")

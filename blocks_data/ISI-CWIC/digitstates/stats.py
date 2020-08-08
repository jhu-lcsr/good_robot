import os,sys

vals = []
for line in open("images_arrays.txt",'r'):
  line = line.split(",")
  v = 0 
  for e in line:
    if int(e) > 0:
      v += 1
  vals.append(v)

vals.sort()
print "Min ", min(vals)
print "Max ", max(vals)
print "Ave ", 1.0*sum(vals)/len(vals)
print "Med ", vals[len(vals)/2]

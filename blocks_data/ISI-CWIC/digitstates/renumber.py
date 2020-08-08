import os,sys,json

def logo(i):
  r = i/10
  c = i - 10*r
  if (r+c)%2 == 0:
    return "logo"
  return "digit"

def label(i):
  if i%10 == 7:
    return "Dev"
  if i%10 >= 8:
    return "Test"
  return "Train"

L = []
D = {}
digit = 0
for line in open('arr.txt','r'):
  line = line.split()
  L.extend(line)
  for v in line:
    D[v] = digit
  digit += 1

for i in range(len(L)):
  section = label(i)
  sid = i
  fileid = L[i]
  digit  = D[L[i]]
  decoration = logo(i)
  print "%-5s  %2d  %3s  %3s  %s" % (label(i), i, L[i], D[L[i]], logo(i))

  j = json.load(open("unraveledJSONs/" + fileid + ".json",'r'))
  j["block_meta"]["decoration"] = decoration
  f = open("MTurk/%s/%d_Num%d.json" % (section, sid, digit),'w')
  f.write(json.dumps(j) + "\n")
  f.close()

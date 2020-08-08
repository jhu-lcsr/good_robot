import os,sys,json

# Set of brands for labeling blocks
brands = [ \
'adidas', 'bmw', 'burger king', 'coca cola', 'esso',  \
'heineken', 'hp', 'mcdonalds', 'mercedes benz', 'nvidia',  \
'pepsi', 'shell', 'sri', 'starbucks', 'stella artois',  \
'target', 'texaco', 'toyota', 'twitter', 'ups'];

# Check the number of non-zero (white) blocks
def size(arr):
  c = 0
  for coord in range(len(arr)):
    c += 1 if int(array[coord]) > 0 else 0
  return c

# We assume all blocks are the same size and shape
shape = {"type": "cube","size": 0.5,"shape_params": { \
          "side_length": "0.1524",  \
          "face_1": {"color": "blue","orientation": "1"}, \
          "face_2": {"color": "green","orientation": "1"}, \
          "face_3": {"color": "cyan","orientation": "1"}, \
          "face_4": {"color": "magenta","orientation": "1"},\
          "face_5": {"color": "yellow","orientation": "1"}, \
          "face_6": {"color": "red","orientation": "2"}}}

# Read in the ids for the images.
ids = []
for line in open("images_ids.txt",'r'):
  ids.append(line.split(".png")[0])

# Read one image array per line
arrays = []
for line in open("images_arrays.txt",'r'):
  arrays.append(line.strip().split(","))

# For each image
for i in range(len(ids)):
  imageID = ids[i]
  array   = arrays[i]
  if size(array) <= 20:
    uniq_id = 0
    f = open("JSONs/%s.json" % imageID,'w')
    j = {"block_meta":{"blocks":[]}}

    # Give each white square a block (and brand)
    for coord in range(len(array)):
      if int(array[coord]) > 0:
        j["block_meta"]["blocks"].append( {"name":brands[uniq_id], "id":(uniq_id+1), "shape": shape})
        uniq_id += 1

    # Assign a location to each block
    uniq_id = 0
    j["block_state"] =  []
    for coord in range(len(array)):
      r = coord / 14
      c = coord % 14
      if int(array[coord]) > 0:
        j["block_state"].append({"id":(uniq_id+1), "position":"%f,0.1,%f" % (2.0/3*(r - 6)/4, -2.0/3*(c - 6)/4)})
        uniq_id += 1
    f.write(json.dumps(j))
    f.close()

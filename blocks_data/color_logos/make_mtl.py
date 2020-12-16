import sys 
HEADER = """newmtl CubeTop
Kd 1 1 1
Ns 96.0784
d 1
illum 1
Ka 0 0 0
Ks 1 1 1
map_Kd {}.png"""

logo_type = sys.argv[1]
with open(f"{logo_type}.mtl", "w") as f1:
    f1.write(HEADER.format(logo_type))

'''

Test for the PrimeSense Sensor for the Real Good Robot project
Author: Hongtao Wu
Sep 29, 2019

'''

from primesense_sensor import PrimesenseSensor
import cv2

def main():
    PS = PrimesenseSensor()
    PS.start()

    color_img, depth_img, _ = PS.frames()
#    print(f'color_img type: {color_img.raw_data.dtype}')
#    print(f'depth_img type: {depth_img.raw_data.dtype}')

    color_img_raw = cv2.cvtColor(color_img.raw_data, cv2.COLOR_BGR2RGB)

    cv2.imshow('test_color.png', color_img_raw)
    cv2.imshow('test_depth.png', depth_img.raw_data)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()

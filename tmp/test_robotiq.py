# https://blog.robotiq.com/controlling-the-robotiq-2f-gripper-with-modbus-commands-in-python

import serial
import time
import binascii

ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1,
                    parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
counter = 0

while counter < 1:
    counter = counter + 1
    ser.write("\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
    data_raw = ser.readline()
    print('Raw data', data_raw)
    data = binascii.hexlify(data_raw)
    print("Response 1 ", data)
    time.sleep(0.01)

    ser.write("\x09\x03\x07\xD0\x00\x01\x85\xCF")
    data_raw = ser.readline()
    print('Raw data', data_raw)
    data = binascii.hexlify(data_raw)
    print("Response 2 ", data)
    time.sleep(1)

while(True):
    print("Close gripper")
    ser.write("\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29")
    data_raw = ser.readline()
    print('Raw data', data_raw)
    data = binascii.hexlify(data_raw)
    print("Data Response 3 ", data)
    time.sleep(2)

    print("Open gripper")
    ser.write("\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
    data_raw = ser.readline()
    print('Raw data', data_raw)
    data = binascii.hexlify(data_raw)
    print("Response 4 ", data)
    time.sleep(2)

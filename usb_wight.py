import serial
import time

ser = serial.Serial(
     port='COM3',
     baudrate = 9600,
     parity=serial.PARITY_NONE,
     stopbits=serial.STOPBITS_ONE,
     bytesize=serial.EIGHTBITS,
)

while 1:
    x=ser.readline()
    print(x)
    time.sleep(1)
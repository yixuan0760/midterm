import numpy as np
import serial
import time

waitTime = 0.1

# generate the waveform table
signalLength = [42, 34, 32]
signalTable = [[261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261],
  [261, 294, 330, 349, 392, 440, 349, 330, 294, 261,
  392, 349, 330, 392, 349, 330, 294,
  392, 349, 330, 392, 349, 330, 294, 
  261, 294, 330, 349, 392, 440, 349, 330, 294, 261,],
  [261, 294, 330, 261, 261, 294, 330, 261,
  330, 349, 392, 330, 349, 392,
  392, 440, 392, 349, 330, 261,
  392, 440, 392, 349, 330, 261,
  294, 196, 261, 196, 392, 261]]
noteLength = [[1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2],
  [1, 1, 1, 1, 2, 1, 1, 2, 2, 3,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 2, 1, 1, 2, 2, 3],
  [1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 2, 2,
  1, 1, 1, 1, 2, 2,
  1, 1, 2, 1, 1, 2]]

# output formatter
formatter = lambda x: "%.3f" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
while(1):
  
  line=s.readline() # Read an echo string from K66F terminated with '\n'
 
  # print line
  
  index = line[0]-48
  
  print(index)

  print("Sending signal ...")

  print("It may take about %d seconds ..." % (int(signalLength[index] * waitTime * 2)))
  
  for data in signalTable[index]:
    
    s.write(bytes(formatter(data), 'UTF-8'))
    
    time.sleep(waitTime)
 
  for data in noteLength[index]:
    
    s.write(bytes(formatter(data), 'UTF-8'))
    
    time.sleep(waitTime)
  
  print("Signal sended")

s.close()
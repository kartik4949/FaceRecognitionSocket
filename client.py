'''from __future__ import print_function
import requests
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

addr = 'http://localhost:5000'
test_url = addr + '/'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('test.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
#print(np.asarray(json.loads(response.content)).shape)
#print(np.asarray(json.loads(response.text)))

plt.imshow(np.asarray(json.loads(response.content)))
plt.show()'''

import cv2
import numpy as np
import socket
import sys
import _pickle as cPickle
import struct
import multiprocessing

BUFFER_SIZE = 4096*4


def sender():
    cap=cv2.VideoCapture(0)
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect(('3.134.97.106',8081)) 
    data = b'' 
    payload_size = struct.calcsize("L")
    
    while True:
        ret,frame=cap.read()
        
        sent_data = cPickle.dumps(frame)
        message_size = struct.pack("L", len(sent_data)) ### CHANGED
        clientsocket.sendall(message_size + sent_data)
        while len(data) < payload_size:
            data += clientsocket.recv(BUFFER_SIZE)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += clientsocket.recv(BUFFER_SIZE)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = cPickle.loads(frame_data)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    print('done')

    clientsocket.close()
'''
def reciever():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    conn, addr = s.accept()
    data = b'' 
    payload_size = struct.calcsize("L") 
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
'''

if __name__ == '__main__':
    
    p1 = multiprocessing.Process(target=sender)
    #p1 = multiprocessing.Process(target=reciever)
    

    p1.start()
    p1.join()

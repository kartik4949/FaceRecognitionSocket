'''from flask import Flask,request,Response
import numpy as np
import cv2
from flask_cors import CORS, cross_origin
import threading

import jsonpickle
import json
import matplotlib.pyplot as plt
from flask import send_file
def face_detect(img):
    with Face_recognition("test") as fr:
        return fr.multi_detectFace(img)


app = Flask(__name__)



@app.route('/',methods=['POST'])
def face():
    exec_error = None
    try:
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img = img[:,:,[2,1,0]]        
        #t1 = threading.Thread(target=face_detect, args=(img)) 
        #t1.start() 
        #t1.join() 
        print("Done!")
    except IOError:
        print('An error occured trying to read the file.')
    
    except ValueError:
        print('Non-numeric data found in the file.')

    except ImportError:
        print("NO module found")
        
    except EOFError:
        print('Why did you do an EOF on me?')

    except KeyboardInterrupt:
        print('You cancelled the operation.')

    except :
        print('An error occured.')
    else:

        img = face_detect(img)
        plt.imshow(img)
        plt.show()
        def default(obj):
            if type(obj).__module__ == np.__name__:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj.item()
            raise TypeError('Unknown type:', type(obj))
        
        encode response using jsonpickle
        response_pickled = jsonpickle.encode(face_detect(img))
        response_pickled = json.dumps(img,default = default)

        
        return Response(response=response_pickled, status=200, mimetype="application/json") 
        #return send_file(img)      
        

        

if __name__ == '__main__':
    cors = CORS(app, resources={r"/*": {"origins": "*"}})
    app.run(host='0.0.0.0',debug=True)

    '''


import _pickle as cPickle
import socket
import struct
from face_recognition_final_v2 import Face_recognition
import cv2
import multiprocessing
    
HOST = ''
PORT = 8081
BUFFER_SIZE  = 4096*4

def reciever():
    fr = Face_recognition('test')
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
        data = b'' 
        while len(data) < payload_size:
            data += conn.recv(BUFFER_SIZE)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(BUFFER_SIZE)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        #data = data + frame_data
        frame = cPickle.loads(frame_data)
        print(frame.shape)
        frame = fr.multi_detectFace(frame)
        data = cPickle.dumps(frame)
        message_size = struct.pack("L", len(data)) ### CHANGED
        conn.sendall(message_size + data)
        #child_pipe.send(frame)
        #cv2.imshow('frame', frame)
        #cv2.waitKey(1)
    conn.close()
    del fr
'''
def sender(parent_pipe):
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    counter = 0
    while counter < 100:
        try:
            conn = clientsocket.connect(('172.16.16.137',8089))

        except socket.error as error:    
           print("Connection Failed **BECAUSE:** {}".format(error))
           print("Retrying to connect....... ||Attempt {} of 100||".format(counter))
           counter += 1
        else:
            break
    

    while True:
        #ret,frame=cap.read()
        frame =  parent_pipe.recv()
        data = pickle.dumps(frame)
        message_size = struct.pack("L", len(data)) ### CHANGED
        clientsocket.sendall(message_size + data)
    clientsocket.close()
'''

if __name__ == '__main__':


    #child_pipe,parent_pipe = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=reciever)
    #p2 = multiprocessing.Process(target=sender,args=(parent_pipe,))
    p1.start()
    #p2.start()
    p1.join()
    #p2.join()

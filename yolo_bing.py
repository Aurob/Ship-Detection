#Imports for handling requests
import time
import socket
import json


#Urllib for downloading images
#OpenCV and NumPy for image processing
import urllib.request as urlr 
import cv2, numpy as np 

#Detection can take some time depending on the image size
#Use threads to allow multiple processes to occur at once
from _thread import *
import threading


#preload opencv DNN object
weight = 'yoloV3_3kplus.weights'
config = 'yoloV3.cfg'
layers = []
net = cv2.dnn.readNet(weight, config)
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Helper function for simplyfing string to byte conversion
def as_bytes(data):
    return bytes(data, encoding="utf-8")


#Server class for handling and processing all requests
class server:
    
    #__Initialize values and create the thread lock object
    def __init__(self, ip, port):
        self.HOST = ip
        self.PORT = int(port)
        self.BUFFER = 51200
        self.thread_lock = threading.Lock()

    #__Start the server, handle valid api requests only
    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            #tells the kernel to reuse a local socket in TIME_WAIT state
            #without waiting for its natural timeout to expire.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.HOST, self.PORT))
            s.listen()
            
            print("Server running on: " + self.HOST + ":" + str(self.PORT))
            while 1:
                conn, addr = s.accept()
                data = conn.recv(self.BUFFER).decode('utf-8')
                
                try:
                    #send an initial OK response
                    #set the repsonse type
                    conn.send(b'HTTP/1.0 200 OK\r\n')
                    conn.send(b'Content-Type: application/json\r\n\r\n')
                    
                    #start a new thread for the processing to occur on
                    self.thread_lock.acquire()
                    start_new_thread(self.handle_query, (data,conn,))
                    
                except Exception as e:
                    print(e)
                    self.error(conn, 'unidentified error')

    #__Helper function for sending valid requests
    def send(self, conn, retval):
        conn.send(retval)
        #close connection
        conn.close()

    #__Helper function used to send an error JSON object if any error occurs
    def error(self, conn, e):
        #if a client closes the connection before a response is sent
        #conn.send will fail and break the program, so simply close the connection
        if isinstance(e, ConnectionAbortedError):
            conn.close()
        else:
            self.send(conn, as_bytes(json.dumps([{"error": str(e)}, time.time_ns()])))
            
    def handle_query(self, data, conn):
        retval = {}
        try:
            #self.BASE will be the processing endpoint
            #any other endpoint will send the HTML file
            request_type = data.split("\n")[0]
            print(request_type)
            
            #the data sent from the client begins with 'DATA:'
            #split by this value to separate the header and data
            data = data.split("DATA:")[1]

            if data == "HTTP Error 401: Unauthorized":
                self.error(conn, "API Key Invalid")
            #convert the request to a json object
            post = json.loads(data)

            #request data
            key = "AlVUTrENvRYkOVi5K3PVdyxritBA0U6Vt2hfTuqhBVqOaGo9PRDfckJN94LRjQD1" #post['key']
            data = post['data']+"&key="+key
            #obviously 1234 isn't a secure key
            #this is more of a demonstration of using a key to validate requests 
            #if key != '1234':    
            #    raise Exception('Invalid key')

            #'data' will be a Bing Static Maps url
            #download the url as an image
            #then run YOLO dnn processing with the selected weight
            retval = self.process(data)
            self.send(conn, retval)

        #Catch any errors that occur in the endpoint methods
        except Exception as e:
            print(e)
            self.error(conn, e)
            
        #End thread
        self.thread_lock.release()

    def process(self, data):
        print("Processing image...")
        #downloads an image from a url and converts it to a numpy array
        img = self.url_to_img(data)
        height, width, _ = img.shape

        #begin Yolo detection
        #returns top left (x, y), width and height, center (x, y), and confidence
        centers = self.detect(img, width, height)
        res = as_bytes(str(centers)+"\n")
        return res

    def url_to_img(self, url):
        imread = urlr.urlopen(url)
        image = np.asarray(bytearray(imread.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def detect(self, img, width, height):
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(outputlayers)

        class_ids=[]
        confidence=[]
        centers=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.9:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    centers.append({'x': x, 'y': y, 'w': w, 'h': h, 'cx': center_x, 'cy': center_y, 'conf': confidence})

        #include the current time as a key
        centers.append(str(time.time_ns()))

        return centers

if __name__ == '__main__':
    ip = 'localhost'
    port = '10001'
    s = server(ip, port).start_server()

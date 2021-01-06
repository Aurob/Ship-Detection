#Imports for handling requests
import time
import socket
import json
import sys 
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


def process(data):
    #downloads an image from a url and converts it to a numpy array
    img = url_to_img(data)
    height, width, _ = img.shape

    #begin Yolo detection
    #returns top left (x, y), width and height, center (x, y), and confidence
    centers = detect(img, width, height)

    res = str(centers)
    return res

def url_to_img(url):
    imread = urlr.urlopen(url)
    image = np.asarray(bytearray(imread.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def detect(img, width, height):
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



if len(sys.argv) > 1:
    #request data
    key = "AlVUTrENvRYkOVi5K3PVdyxritBA0U6Vt2hfTuqhBVqOaGo9PRDfckJN94LRjQD1" #post['key']
    data = sys.argv[1]+"&key="+key
    
    res = process(data)
    
    sys.stdout.write(res)
        

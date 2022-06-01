import numpy as np
import cv2
import pickle 
from cv2 import VideoCapture
from cv2 import waitKey
from keras.models import load_model

framewidth = 640
frameHeight = 480
brightness = 180
threshold = 0.90
dim = (framewidth,frameHeight)
font = cv2.FONT_HERSHEY_SIMPLEX

#setup video camera
cap = cv2.VideoCapture(2)
cap.set(3, framewidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
model = load_model("C:/Users/dlrjs/openCV/my_model.h5")
model.summary()

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getCalssName(classNo):
    if classNo == '0': 
        return 'speed Limit 20km/h'
    elif classNo == '1': 
        return 'speed Limit 40km/h'
    elif classNo == '2': 
        return 'speed Limit 50km/h'
    elif classNo == '3': 
        return 'speed Limit 60km/h'
    elif classNo == '4': 
        return 'speed Limit 70km/h'
    elif classNo == '5': 
        return 'speed Limit 80km/h'
    elif classNo == '6': 
        return 'end of speed Limit 80km/h'
    elif classNo == '7': 
        return 'speed Limit 100km/h'
    elif classNo == '8': 
        return 'speed Limit 120km/h'
    elif classNo == '9': 
        return 'No passing'
    elif classNo == '10':
        return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == '11': 
        return 'Right-of-way at the next intersection'
    elif classNo == '12': 
        return 'prioirty road'
    elif classNo == '13': 
        return 'Yield'
    elif classNo == '14': 
        return 'Stop'
    elif classNo == '15': 
        return 'No vehicles'
    elif classNo == '16':
        return 'Veh > 3.5 tons prohibited'
    elif classNo == '17':
        return 'No entry'
    elif classNo == '18':
        return 'General caution'
    elif classNo == '19':
        return 'Dangerous curve left'
    elif classNo == '20': 
        return 'Dangerous curve right'
    elif classNo == '21':
        return 'Double curve'
    elif classNo == '22':
        return 'Bumpy road'
    elif classNo == '23':
        return 'Slippery road'
    elif classNo == '24':
        return 'Road narrows on the right'
    elif classNo == '25': 
        return 'Road work'
    elif classNo == '26':
        return 'Traffic signals'
    elif classNo == '27': 
        return 'Pedetrians'
    elif classNo == '28':
        return 'Children crossing'
    elif classNo == '29':
        return 'Bicycles crossing'
    elif classNo == '30':
        return 'Beware of ice/snow'
    elif classNo == '31':
        return 'Wild animals crossing'
    elif classNo == '32':
        return 'End speed + passing limits'
    elif classNo == '33':
        return 'Turn right ahead'
    elif classNo == '34':
        return 'Turn left ahead'
    elif classNo == '35':
        return 'Ahead only'
    elif classNo == '36':
        return 'Go straight or right'
    elif classNo == '37':
        return 'Go straight or left'
    elif classNo == '38':
        return 'keep right'
    elif classNo == '39':
        return 'keep left'
    elif classNo == '40':
        return 'Roundabout mandatory'
    elif classNo == '41':
        return 'End of no passing'
    elif classNo == '42':
        return 'End no passing veh > 3.5 tons'

while True:
 
# READ IMAGE
    success,resized = cap.read()

    img = np.asarray(resized)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(resized, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(resized, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions[0])
    probabilityValue =np.amax(predictions)
   
    #if probabilityValue > threshold:
    #    cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA) 
    #    cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)        
    #cv2.imshow("Result", imgOrignal)
    
    gray = cv2.medianBlur(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY),5)
    resized = cv2.resize(gray,dim,interpolation = cv2.INTER_AREA)
    edges = cv2.Canny(gray,100,200)
    circ = cv2.HoughCircles(resized,cv2.HOUGH_GRADIENT,1,30,param1=50,param2=75,
                              minRadius=0,maxRadius=0)
    if probabilityValue > threshold:
        cv2.putText(resized,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA) 
        cv2.putText(resized,str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
 
    cv2.imshow('video',resized)
    if circ is not None:
        circ = np.uint16(np.around(circ))[0,:]
        print(circ)
        for j in circ:
            cv2.circle(resized, (j[0], j[1]), j[2], (0, 255, 0), 2)
        cv2.imshow('video',resized)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
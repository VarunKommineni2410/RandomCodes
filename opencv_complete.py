#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[3]:


import cv2
import random
import pandas as pd
import numpy as np


# In[16]:


#reading and resizing
#-1,0,1 represent different scales

img=cv2.imread("logo.jpg",1)
img=cv2.resize(img,(450,500))
#img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
print(img.shape)

#opening and closing image

cv2.imshow("one",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


#rotate

img=cv2.rotate(img,cv2.ROTATE_180)
cv2.imwrite("new.jpg",img)

cv2.imshow("one",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[36]:


print(img.shape)
print()

print(img.shape[0]) #reprents row
print(img.shape[1]) #reprents column
print(img.shape[2]) #reprents channels


# In[34]:


#manpulating image

img=cv2.imread("logo.jpg",1)
img=cv2.resize(img,(450,500))

for i in range(50,100):
    for j in range(img.shape[1]):
        img[i,j]=[random.randrange(255),random.randrange(255),random.randrange(255)]

        
cv2.imshow("one",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[35]:


#copy and pasting a part of a image
print(img.shape)
new=img[50:100,250:300]
img[350:400,400:450]=new

cv2.imshow("one",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[39]:


#Camera 

cap=cv2.VideoCapture(0) #open camera (0-random cam,1-first cam etc)

while True:
    
    ret,frame=cap.read() #reading frame in cam
    cv2.imshow("cam",frame) #showing cam
    
    if cv2.waitKey(1)==ord("v"): #closing if v(ASCII) value is matched checked every 1 milliseond
        break
        
cap.release() # closing cam
cv2.destroyAllWindows() 


# In[29]:



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3)) # 3 is identifier for width
    height = int(cap.get(4)) # 4 is identifier for height

    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, width//2:] = smaller_frame

    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[41]:


#line drawing

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    cv2.line(frame,(0,0),(width,height),[0,0,255],8)  #line(image,start,end,color,thickness)
    
    cv2.imshow('camera', frame)

    if cv2.waitKey(1) == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()


# In[49]:


#cross line

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    lm= cv2.line(frame,(0,0),(width,height),[0,0,255],8)  #line(image,start,end,color,thickness)
    lm= cv2.line(frame,(0,height),(width,0),[255,0,255],8)  #anotherline
    
    cv2.imshow('camera', lm)

    if cv2.waitKey(1) == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()


# In[52]:


#Drawing shapes and text

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 10)
    img = cv2.line(img, (0, height), (width, 0), (0, 255, 0), 5)
    
    img = cv2.rectangle(img, (100, 100), (200, 200), (128, 128, 128), 5) #rectangle(image,start,end,color,thickness)
    
    img = cv2.circle(img, (300, 300), 60, (0, 0, 255), -1) #circle(image,center,radius,color,thickness(-ve will fill that shape))
    
    font = cv2.FONT_HERSHEY_SIMPLEX # creating font
    
    img = cv2.putText(img, 'HELLO', (10, height - 10), font, 4, (0, 0, 0), 5, cv2.LINE_AA) #putText(img,text,center_pos,font,font_scale,color,line_thick,line_type)

    cv2.imshow('frame', img)


    if cv2.waitKey(1) == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()


# In[23]:


#HSV, taking out only orange color in camera

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
   
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #cvColor(img,into_hsv_image)
    
   
    low_blue=np.array([0, 164, 0]) #bound between diff orange
    upper_blue=np.array([130, 255, 255])
    
    mask=cv2.inRange(hsv,low_blue,upper_blue) #to have the color in range (1-true,0-false)
    
    result=cv2.bitwise_and(frame,frame,mask=mask) #comparing images and giving only the blue color 
    
    cv2.imshow('normal', frame) #normal img
    cv2.imshow('frame', result) #only orange shades are highlighte
    cv2.imshow('mask', mask) #selected is white and not selected is black
    
    if cv2.waitKey(1) == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()


# In[29]:


#HSV, taking out only orange color in photo

#cap = cv2.VideoCapture(0)

#ret, frame = cap.read()
frame=cv2.imread("orange.jpg")
#width = int(cap.get(3))
#height = int(cap.get(4))

hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #cvColor(img,into_hsv_image)
    
   
low_blue=np.array([0, 164, 0]) #bound between diff orange
upper_blue=np.array([130, 255, 255])
    
mask=cv2.inRange(hsv,low_blue,upper_blue) #to have the color in range (1-true,0-false)
    
result=cv2.bitwise_and(frame,frame,mask=mask) #comparing images and giving only the blue color 
    
cv2.imshow('normal', frame) #normal img
cv2.imshow('frame', result) #only orange shades are highlighte
cv2.imshow('mask', mask) #selected is white and not selected is black
    
#if cv2.waitKey(1) == ord('k'):
       # break
    
cv2.waitKey(0) == ord('k')
cap.release()
cv2.destroyAllWindows()


# In[37]:


#incomplete

import numpy as np
import cv2

img = cv2.imread('box.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
	x, y = corner.ravel()
	cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

for i in range(len(corners)):
	for j in range(i + 1, len(corners)):
		corner1 = tuple(corners[i][0])
		corner2 = tuple(corners[j][0])
		color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
		cv2.line(img, corner1, corner2, color, 1)

cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# incomplete

import numpy as np
import cv2

img = cv2.resize(cv2.imread('assets/soccer_practice.jpg', 0), (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(cv2.imread('assets/shoe.PNG', 0), (0, 0), fx=0.8, fy=0.8)
h, w = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)    
    cv2.rectangle(img2, location, bottom_right, 255, 5)
    cv2.imshow('Match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[38]:


# incomplete

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


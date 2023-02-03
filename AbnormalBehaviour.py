from tkinter import *
import tkinter
import numpy as np
import imutils
import dlib
import h5py
import cv2 as cv
from gtts import gTTS
import pyttsx3
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing import image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

main = tkinter.Tk()
main.title("DETECTION OF ANOMALOUS DRIVER BEHAVIOUR AND ALERTING USING DEEP LEARNING")
main.geometry("800x500")

global model1
global video
class IntimatingDriver:
    messagetointimate=" "
    @staticmethod
    def text_to_speech(s2):
        messagetointimate=s2
        gttsobj=gTTS(text=messagetointimate,lang='en',slow=False)
        gttsobj.save("welcome.mp3")
        os.system("welcome.mp3")
        return
speech=IntimatingDriver()
    
class Model:
 @staticmethod
 def loadModel():
    global model1
    image_width, image_height = 150, 150
    if os.path.exists('model.h5'):
        model1 = Sequential()
        model1.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3), activation='relu'))
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Convolution2D(32, 3, 3, activation='relu'))
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Flatten())
        model1.add(Dense(output_dim=128, activation='relu'))
        model1.add(Dense(output_dim=10, activation='softmax'))
        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model1.load_weights('model.h5')
        print(model1.summary())
        pathlabel.config(text="Model Generated Successfully")
   
model=Model()

class UserInterface:
 @staticmethod
 def upload():
    global video
    filename = filedialog.askopenfilename(initialdir="Video")
    pathlabel.config(text="          Video loaded")
    video = cv.VideoCapture(filename)

 @staticmethod
 def startMonitoring():
     i=0
     seconds=60
    
     """ret, frame = video.read()
        fps=video.get(cv.CAP_PROP_FPS)
        length_minutes=3
        frames1=length_minutes*60*fps
        n=40
        desired_frames=n*np.arange(frames1)
        for i in desired_frames:
            video.set(1,i-1)
            ret,frame=video.read(1)
            frameId=video.get(1)"""
 
     while (True):
          ret, frame = video.read()
          print(ret)
          if ret == True:
             cv.imwrite("test.jpg", frame)
             imagetest = image.load_img("test.jpg", target_size=(150, 150))
             imagetest = image.img_to_array(imagetest)
             imagetest = np.expand_dims(imagetest, axis=0)
             predict = model1.predict_classes(imagetest)
             print(predict)
             msg =""
             if str(predict[0]) == '0':
              msg = 'makeup'
             if str(predict[0]) == '1':
              msg = 'Using/Talking Phone'
              s1='Please Stop using phone while driving'
              IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '2':
                  msg = 'Talking On phone'
                  s1='Please stop using phone while driving'
                  IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '3':
                  msg = 'Using/Talking Phone'
                  s1='please stop using phone while driving'
                  IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '4':
                  msg = 'Using/Talking Phone'
                  s1='please stop using phone while driving'
                  IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '5':
                  msg = 'Radio Operating'
                  s1='please stop operating radio while driving'
                  IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '6':
                   msg = 'Drinking'
                   s1='please stop drinking while driving'
                   IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '7':
                   msg = 'Reaching Behind'
                   s1='please stop reaching behind while driving'
                   IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '8':
                   msg = 'Hair & Makeup'
                   s1='please stop doing hair and makeup while driving'
                   IntimatingDriver.text_to_speech(s1)
             if str(predict[0]) == '9':
                   msg = 'Talking To Passenger'
                   s1='please stop talking to passenger while driving'
                   IntimatingDriver.text_to_speech(s1)
            
             text_label = "{}: {:4f}".format(msg, 80)
             cv.putText(frame, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
             cv.imshow('Frame', frame)
             if cv.waitKey(25000) & 0xFF == ord('q'):
                  break
          else:
               break
     video.release()
     cv.destroyAllWindows()
    
    
def exit():
    global main
    main.destroy()
application=UserInterface()

font = ('times', 16, 'bold')
title = Label(main, text='DETECTION OF ANOMALOUS DRIVER BEHAVIOUR AND ALERTING USING DEEP LEARNING', anchor=W,justify=LEFT)
title.config(bg='black', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
loadButton = Button(main, text="Generate & Load Model", command=Model.loadModel)
loadButton.place(x=50, y=200)
loadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(fg='black')
pathlabel.config(font=font1)
pathlabel.place(x=50, y=250)

uploadButton = Button(main, text="Upload Video", command=UserInterface.upload)
uploadButton.place(x=50, y=300)
uploadButton.config(font=font1)

uploadButton = Button(main, text="Start Behaviour Monitoring", command=UserInterface.startMonitoring)
uploadButton.place(x=50, y=350)
uploadButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50, y=400)
exitButton.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()




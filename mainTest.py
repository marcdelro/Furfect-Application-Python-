from tkinter import *
from tkinter import filedialog
from keras.models import load_model
from keras.applications import imagenet_utils
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
import webbrowser

classes = os.listdir('train/')
globalBird = ''

def openFileDialog():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image", filetypes=(("JPG File", "*.jpg"), ("PNG FIle", "*.png")))

    if(filename != ""):
        img = Image.open(filename)
        img = img.resize((350, 350))
        img = ImageTk.PhotoImage(img)
        label.configure(image=img)
        label.image = img
        predictImage(filename)

def predictImage(path):
    global globalBird
    INPUT_SIZE = 256
    image_pred = cv2.imread(path)
    img = Image.fromarray(image_pred)
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img = np.array(img)

    img_input = np.expand_dims(img, axis=0)

    result = np.argmax(model.predict(img_input, verbose=2), axis=1)
    resultText.configure(text="This is a/n {}".format(classes[int(result)].capitalize()))
    globalBird = classes[int(result)].capitalize()
    print(model.predict(img_input))
    print(np.argmax(model.predict(img_input), axis=-1))
    openLinkBtn.pack(side=tk.RIGHT, padx=20)

def displayLink():
    global globalBird
    webbrowser.open('https://en.wikipedia.org/wiki/{}'.format(globalBird))


#Frame
root = Tk()
frame = Frame(root)
frame.pack(side=BOTTOM, padx=15, pady=15)

root.resizable(0, 0)

label = Label(root)
label.place(x=250, y=200, anchor="center")

resultText = Label(root)
resultText.pack(side=BOTTOM)

selectImgBtn = Button(frame, text="Select Image", command=openFileDialog)
selectImgBtn.pack(side=tk.LEFT, padx=20)

openLinkBtn = Button(frame, text="Open Wikipedia Article", command=displayLink)

model = load_model('DogBreedModel.h5', custom_objects={'imagenet_utils': imagenet_utils})

root.title("Dog Breed Classification")
root.geometry("500x500")
root.mainloop()
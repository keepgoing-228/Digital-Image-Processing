import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import tkinter
from tkinter.constants import BOTH, LEFT, X, Y
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

from numpy.matrixlib import matrix



class grayReader(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Gray Reader')
        self.geometry('1000x700+100+50')
        self.resizable(0,0)
        self.config(background="LightCyan")
        self.setup()

        
    def setup(self):
        self.inputBtn1 = tk.Button(self, text='Input First File',command=self.showFirstBase64) # if there is () it will run function first
        self.inputBtn1.place(x=100, y=1)
        self.Histogram1 = tk.Button(self, text='Show First Histogram', command=lambda:self.showHistogram(self.counting1))
        self.Histogram1.place(x=200, y=1) 

        self.inputBtn2 = tk.Button(self, text='Input Second File', command=self.showSecondBase64)
        self.inputBtn2.place(x=480, y=1)
        self.Histogram1 = tk.Button(self, text='Show Second Histogram', command=lambda:self.showHistogram(self.counting2))
        self.Histogram1.place(x=600, y=1)

        self.addLabel = tk.Label(self, text='First file add or subtract:', font=15)
        self.addLabel.place(x=100, y=200)
        self.changedAddValue = tk.IntVar()
        self.changedAddValue.set(0)
        self.addCombobox = ttk.Combobox(self, values=[-60, -50,-40,-30,-20,-10,0,10,20,30,40,50] ,textvariable = self.changedAddValue, state="readonly")
        self.addCombobox.place(x=300, y=200)
        self.addBtn = tk.Button(self, text='add result of the first file', command=self.add)
        self.addBtn.place(x=100, y=230)

        self.multiLabel = tk.Label(self, text='First file multiply:', font=15)
        self.multiLabel.place(x=500, y=200)
        self.changedMultiValue = tk.IntVar()
        self.changedMultiValue.set(1)
        self.multiCombobox = ttk.Combobox(self, values=[1,2,3,4,5] ,textvariable = self.changedMultiValue, state="readonly")
        self.multiCombobox.place(x=650, y=200)
        self.multiBtn = tk.Button(self, text='multiply result of the first file', command=self.multi)
        self.multiBtn.place(x=500, y=230)

        self.averageBtn = tk.Button(self, text='average', command=self.average)
        self.averageBtn.place(x=100, y=450)
        self.averageHistogram = tk.Button(self, text='Show average Histogram', command=lambda:self.showHistogram(self.averageCounting))
        self.averageHistogram.place(x=180, y=450)
        self.fBtn = tk.Button(self, text='function', command=self.function)
        self.fBtn.place(x=500, y=450)
        self.fHistogram = tk.Button(self, text='Show function Histogram', command=lambda:self.showHistogram(self.fCounting))
        self.fHistogram.place(x=580, y=450)

        # show full np matrix
        np.set_printoptions(threshold=np.inf)


    def readBase64(self, matrix, count, counting):
        fileName = filedialog.askopenfilename()
        try:
            with open(fileName,"rb") as base64File: #Base64 encoding is a type of conversion of bytes into ASCII characters.
                for j in range(64):
                    line = base64File.readline()
                    for i in range(64):
                        #auto convert string to ascii
                        if (line[i])>64:  # range from A to Z and a to z
                            matrix[j,i] = ((line[i] - 55) * 8) # A equal to 65 and A means 10
                            count[j,i] = (line[i]-55)
                        else:  #range from 0 to 9
                            matrix[j,i] = ((line[i] - 48) * 8)
                            count[j,i] = (line[i]-48)
                
                for x in range(64):
                    for y in range(64):
                        for z in range(32):
                            if count[x][y] == z:
                                counting[z] = counting[z] +1
    
        except:
            return None         



    def showFirstBase64(self):
        self.matrix1 = np.zeros((64,64))
        self.count1 = np.zeros((64,64))
        self.counting1 = np.zeros(32)
         
        # print(self.matrix1)
        self.readBase64(self.matrix1, self.count1, self.counting1)
        # print("-----after reading ------")
        # print(self.matrix1)

        self.image1 = Image.fromarray(np.uint8(self.matrix1))
        self.image1 = self.image1.resize((120,120), Image.ANTIALIAS) # remember to return
        imagetk = ImageTk.PhotoImage(self.image1)
        self.image1 = tk.Label(self, image= imagetk, width=120, height=120)
        self.image1.image = imagetk
        self.image1.place(x=100, y=30)
        

    def showSecondBase64(self):
        self.matrix2 = np.zeros((64,64))
        self.count2 = np.zeros((64,64))
        self.counting2 = np.zeros(32)
        self.readBase64(self.matrix2, self.count2, self.counting2)
        self.image2 = Image.fromarray(np.uint8(self.matrix2))
        self.image2 = self.image2.resize((120, 120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.image2)
        self.image2 = tk.Label(self, image= imagetk, width=120, height=120)
        self.image2.image = imagetk
        self.image2.place(x = 480, y=30)

    def showHistogram(self, counting): 
        try:
            plt.bar(range(1,33), counting)
            plt.show()
        except:
            tkinter.messagebox.showwarning('warning','input the pic first')

    def add(self):
        self.addCounting = np.zeros(32)
        # self.addMatrix = self.matrix1 # not pointer
        self.addMatrix = np.zeros((64,64))
        for x in range(64):
            for y in range(64):
                self.addMatrix[x][y] = self.matrix1[x][y]
        for x in range(64):
            for y in range(64):
                self.addMatrix[x][y] = self.addMatrix[x][y] + int(self.changedAddValue.get())
                if self.addMatrix[x][y] > 255:
                    self.addMatrix[x][y] =255
                elif self.addMatrix[x][y] < 0:
                    self.addMatrix[x][y] =0
        self.addCount = self.addMatrix/8

        self.addImage = Image.fromarray(np.uint8(self.addMatrix))
        self.addImage = self.addImage.resize((120, 120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.addImage)
        self.addImage = tk.Label(self, image= imagetk, width=120, height=120)
        self.addImage.image = imagetk
        self.addImage.place(x=100, y= 280)
        for x in range(64):
            for y in range(64):
                for z in range(32):
                    if self.addCount[x][y] == z:
                        self.addCounting[z] = self.addCounting[z] + 1
            
        
    def multi(self):
        self.multiCounting = np.zeros(32)
        self.multiMatrix = np.zeros((64,64))
        for x in range(64):
            for y in range(64):
                self.multiMatrix[x][y] = self.matrix1[x][y]
        for x in range(64):
            for y in range(64):
                self.multiMatrix[x][y] = self.multiMatrix[x][y] * int(self.changedMultiValue.get())
                if self.multiMatrix[x][y] > 255:
                    self.multiMatrix[x][y] = 255
                elif self.multiMatrix[x][y] < 0:
                    self.multiMatrix[x][y] =0 
        # self.multiMatrix = self.multiCount * int(self.changedMultiValue.get())
        self.multiCount = self.multiMatrix /8

        self.multiImage = Image.fromarray(np.uint8(self.multiMatrix))
        self.multiImage = self.multiImage.resize((120, 120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.multiImage)
        self.multiImage = tk.Label(self, image= imagetk, width=120, height=120)
        self.multiImage.image = imagetk
        self.multiImage.place(x=500, y= 280)
        for x in range(64):
            for y in range(64):
                for z in range(32):
                    if self.multiCount[x][y] == z:
                        self.multiCounting[z] = self.multiCounting[z] + 1

    def average(self):
        self.averageCounting = np.zeros(32)
        self.averageMatrix = ((self.matrix1 + self.matrix2) / 2)
        self.averageCount = np.uint8(self.averageMatrix *8)

        self.averageImage = Image.fromarray(self.averageMatrix)
        self.averageImage = self.averageImage.resize((120,120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.averageImage)
        self.ave=tk.Label(self, image=imagetk, width=120, height=120)
        self.ave.image=imagetk
        self.ave.place(x=100,y=500)
        for x in range(64):
            for y in range(64):
                for z in range(32):
                    if self.averageCount[x][y] == z:
                        self.averageCounting[z] = self.averageCounting[z] + 1
                        # print(self.averageCounting)


    def function(self):
        self.fCounting = np.zeros(32)
        self.fMatrix = np.zeros((64,64))
        for x in range(64):
            for y in range(64):
                self.fMatrix[x][y] = self.matrix1[x][y]
        for y in range(64):
            for x in range(64):
                if x==0:
                    self.fMatrix[y][x] = 0
                else:
                    self.fMatrix[y][x] = self.matrix1[y][x] - self.matrix1[y][x-1]
                # check negative values
                if self.fMatrix[y][x] < 0:
                    self.fMatrix[y][x] = 0
        self.fCount = self.fMatrix/8

        self.fImage = Image.fromarray(np.uint8(self.fMatrix))
        self.fImage = self.fImage.resize((120,120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.fImage)
        self.fLabel=tk.Label(self, image=imagetk, width=120, height=120)
        self.fLabel.image=imagetk
        self.fLabel.place(x=500,y=500)
        for x in range(64):
            for y in range(64):
                for z in range(32):
                    if self.fCount[x][y] == z:
                        self.fCounting[z] = self.fCounting[z] + 1



view = grayReader()
view.mainloop()

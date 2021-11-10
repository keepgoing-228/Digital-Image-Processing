import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
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
        self.inputBtn = tk.Button(self, text='Input JPG File',command=self.readJPG) # if there is () it will run function first
        self.inputBtn.place(x=100, y=1)
        self.tranformBtn = tk.Button(self, text='Transform to gray sclae', command=self.RGBtoGray)
        self.tranformBtn.place(x=280, y=1)
        self.compareBtn = tk.Button(self, text='Compare the grayscale images', command=self.substract)
        self.compareBtn.place(x=650, y=1)
        self.thresholdBtn = tk.Button(self, text='Threshold function', command=self.threshold)
        self.thresholdBtn.place(x=280, y=250)
        self.thresholdValue = tk.IntVar()
        self.thresholdValue.set(128)
        self.thresholdCombobox = ttk.Combobox(self, values=[50,100,128,150,200] ,textvariable = self.thresholdValue, state="readonly")
        self.thresholdCombobox.place(x=400, y=250)
        
        self.changedMultiValue = tk.DoubleVar()
        self.changedMultiValue.set(1)
        self.multiCombobox = ttk.Combobox(self, values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
                                        ,textvariable = self.changedMultiValue, state="readonly")
        self.multiCombobox.place(x=700, y=250)
        self.multiBtn = tk.Button(self, text='Adjust brightness:', command=self.multi)
        self.multiBtn.place(x=580, y=250)
        self.equalizationBtn = tk.Button(self, text='equalization', command=self.equlization)
        self.equalizationBtn.place(x=280, y=500)
        # show full np matrix
        np.set_printoptions(threshold=np.inf)

    def readJPG(self):
        self.filename = filedialog.askopenfilename()
        try:
            self.image = Image.open(self.filename).convert('RGB')
            self.image = self.image.resize((120,120), Image.ANTIALIAS) # remember to return
            # to tkinter
            imagetk = ImageTk.PhotoImage(self.image)
            self.imageLabel = tk.Label(self, image= imagetk, width=120, height=120)
            self.imageLabel.image = imagetk
            self.imageLabel.place(x=100, y=60)

        except:
            messagebox.showwarning(title="Warning",message="Please select a JPG file.")



    def RGBtoGray(self):
        # autochanging
        # self.imageGray1 = self.image.convert("L")
        
        # (R+G+B)/3
        self.matrix1 = np.zeros((120,120))
        self.count1 = np.zeros((120,120))
        self.counting1 = np.zeros(32)
        # 0.2989 * R + 0.5870 * G + 0.1140 * B
        self.matrix2 = np.zeros((120,120))
        self.count2 = np.zeros((120,120))
        self.counting2 = np.zeros(32)

        # RGB matrix
        self.Rmatrix = np.zeros((120,120))
        self.Gmatrix = np.zeros((120,120))
        self.Bmatrix = np.zeros((120,120))

        # read to matrix
        R,G,B = self.image.split()
        for y in range(120):
            for x in range(120):
                self.Rmatrix[x,y] = R.getpixel((y,x)) # only  one parameter
                self.Gmatrix[x,y] = G.getpixel((y,x))
                self.Bmatrix[x,y] = B.getpixel((y,x))
        

        # (R+G+B)/3 methods
        self.matrix1 = (self.Rmatrix + self.Gmatrix + self.Bmatrix)/3
        self.matrix1 = self.matrix1.astype(int)
        self.count1 = self.matrix1/8
        for x in range(120):
            for y in range(120):
                for z in range(32): # 32*8 = 256
                    if self.count1[x][y] == z:
                        self.counting1[z] = self.counting1[z] + 1
        
        self.imageGray1 = Image.fromarray(self.matrix1)
        imagetk = ImageTk.PhotoImage(self.imageGray1)
        self.imageGrayLB1 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageGrayLB1.image = imagetk
        self.imageGrayLB1.place(x = 280, y=60)
        self.imageGrayLabel1 = tk.Label(self, text="GRAY = (R+G+B)/3.0")
        self.imageGrayLabel1.place(x=280, y=30)
        self.imageGrayHistogram1 = tk.Button(self, text='Show the first Histogram', command=lambda:self.showHistogram(self.counting1))
        self.imageGrayHistogram1.place(x=280, y=200)


        # use fomula  0.2989 * R + 0.5870 * G + 0.1140 * B methods
        self.matrix2 = 0.2989 * self.Rmatrix + 0.5870 * self.Gmatrix + 0.1140 * self.Bmatrix
        self.matrix2 = self.matrix2.astype(int)
        self.count2 = self.matrix2/8
        for x in range(120):
            for y in range(120):
                for z in range(32):
                    if self.count2[x][y] == z:
                        self.counting2[z] = self.counting2[z] + 1
        
        self.imageGray2 = Image.fromarray(self.matrix2)
        imagetk = ImageTk.PhotoImage(self.imageGray2)
        self.imageGrayLB2 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageGrayLB2.image = imagetk
        self.imageGrayLB2.place(x=460, y=60)
        self.imageGrayLabel2 = tk.Label(self, text="GRAY = 0.299*R + 0.587*G + 0.114*B")
        self.imageGrayLabel2.place(x=410, y=30)
        self.imageGrayHistogram2 = tk.Button(self, text='Show the second Histogram', command=lambda:self.showHistogram(self.counting2))
        self.imageGrayHistogram2.place(x=440, y=200)
        '''
        R = R.point(lambda i: i * 0.3333)
        G = G.point(lambda i: i * 0.3333)
        B = B.point(lambda i: i * 0.3333)
        self.imageGray1 = Image.merge("RGB",(R,G,B))
        '''
    def showHistogram(self, counting): 
        try:
            plt.bar(range(1,33), counting)
            plt.show()
        except:
            tk.messagebox.showwarning('warning','input the pic first')

    def substract(self):
        self.substractMatrix = np.zeros((120,120))
        self.substractCounting = np.zeros(32)
        
        for y in range(120):
            for x in range(120):
                self.substractMatrix[y][x] = self.matrix1[y][x] - self.matrix2[y][x]
                # check negative values
                # if self.substractMatrix[y][x] < 0:
                #     self.substractMatrix[y][x] = 0
        self.substractCount = self.substractMatrix/8

        self.substractImage = Image.fromarray(np.uint8(self.substractMatrix))
        self.substractImage = self.substractImage.resize((120,120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.substractImage)
        self.substractLabel=tk.Label(self, image=imagetk, width=120, height=120)
        self.substractLabel.image=imagetk
        self.substractLabel.place(x=650,y=60)
        for x in range(64):
            for y in range(64):
                for z in range(32):
                    if self.substractCount[x][y] == z:
                        self.substractCounting[z] = self.substractCounting[z] + 1
    
    def threshold(self):
        self.thresholdMatrix = np.zeros((120,120))
        threshold = self.thresholdValue.get()
        for y in range(120):
            for x in range(120):
                self.thresholdMatrix[y][x] = self.matrix2[y][x]
        for y in range(120):
            for x in range(120):
                if self.thresholdMatrix[y][x] > threshold:
                    self.thresholdMatrix[y][x] = 255
                else:
                    self.thresholdMatrix[y][x] = 0
        
        self.thresholdImage = Image.fromarray(np.uint8(self.thresholdMatrix))
        self.thresholdImage = self.thresholdImage.resize((120,120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.thresholdImage)
        self.thresholdLabel=tk.Label(self, image=imagetk, width=120, height=120)
        self.thresholdLabel.image=imagetk
        self.thresholdLabel.place(x=280,y=290)
    
    def enlargeResolution(self):
        self.enlargeResolutionMatrix = np.zeros((239,239))
        # a = x - floor(x)
        # b = b - floor(y)
        # f = a*b*I()


    def multi(self):
        self.multiCounting = np.zeros(32)
        self.multiMatrix = np.zeros((120,120))
        for x in range(120):
            for y in range(120):
                self.multiMatrix[x][y] = self.matrix2[x][y]
        for x in range(120):
            for y in range(120):
                self.multiMatrix[x][y] = self.multiMatrix[x][y] * self.changedMultiValue.get()
                if self.multiMatrix[x][y] > 255:
                    self.multiMatrix[x][y] = 255
                elif self.multiMatrix[x][y] < 0:
                    self.multiMatrix[x][y] =0
        self.multiMatrix = self.multiMatrix.astype(int)
        self.multiCount = self.multiMatrix / 8

        self.multiImage = Image.fromarray(np.uint8(self.multiMatrix))
        self.multiImage = self.multiImage.resize((120, 120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.multiImage)
        self.multiImage = tk.Label(self, image= imagetk, width=120, height=120)
        self.multiImage.image = imagetk
        self.multiImage.place(x=580, y= 290)
        for x in range(120):
            for y in range(120):
                for z in range(32):
                    if self.multiCount[x][y] == z:
                        self.multiCounting[z] = self.multiCounting[z] + 1      
        self.total = 0
        for i in range(32):
            self.total = self.total + self.multiCounting[i]

        self.multiHistogram = tk.Button(self, text='Show the brightness Histogram', command=lambda:self.showHistogram(self.multiCounting))
        self.multiHistogram.place(x=570, y=420)


    def newGrayscale(self, num):
        p = 0
        # MN = 120*120
        for k in range(num):
            p = p + (self.multiCounting[k] /self.total)
        return int((256-1)*p)

    def equlization(self):
        self.newGrayTable = np.zeros(32)
        self.equlizationMatrix = np.zeros((120,120))
        self.equlizationCount = np.zeros((120,120))
        self.equlizationCounting = np.zeros(32)
        for i in range(32):
            self.newGrayTable[i] = self.newGrayscale(i)        
        print(self.newGrayTable)

        for x in range(120):
            for y in range(120):
                for z in range(32):
                    if self.multiCount[x][y] == z:
                        self.equlizationMatrix[x][y] = self.newGrayTable[z]
        self.equlizationCount = self.equlizationMatrix /8
        print("----")
        print(self.equlizationMatrix)
        self.equlizationImage = Image.fromarray(np.uint8(self.equlizationMatrix))
        self.equlizationImage = self.equlizationImage.resize((120, 120), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.equlizationImage)
        self.equlizationImage = tk.Label(self, image= imagetk, width=120, height=120)
        self.equlizationImage.image = imagetk
        self.equlizationImage.place(x=280, y= 530)

        for x in range(120):
            for y in range(120):
                for z in range(32):
                    if self.equlizationCount[x][y] == z:
                        self.equlizationCounting[z] = self.equlizationCounting[z] + 1   
        self.multiHistogram = tk.Button(self, text='Show the equalization Histogram', command=lambda:self.showHistogram(self.equlizationCounting))
        self.multiHistogram.place(x=280, y=660)




view = grayReader()
view.mainloop()

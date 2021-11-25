import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.constants import BOTH, LEFT, X, Y
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt


class grayReader(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Gray Reader')
        self.geometry('1000x700+100+50')
        self.resizable(0,0)
        self.config(background="LightCyan")
        self.setup()

        
    def setup(self):
        # input image
        self.inputBtn = tk.Button(self, text='Input JPG File',command=self.readJPG) # if there is () it will run function first
        self.inputBtn.place(x=100, y=1)
        # lowpass 
        self.nValue = tk.IntVar()
        self.nValue.set(3)
        self.nCombobox = ttk.Combobox(self, values=[3,5,7,9,11,13,15,17,19,21] ,textvariable = self.nValue, state="readonly")
        self.nCombobox.place(x=485, y=5)
        self.showGuassionLowpassBtn = tk.Button(self, text='Show after guassian lowpass mask', command=self.showGuassianLowpass)
        self.showGuassionLowpassBtn.place(x=280, y=1)
        self.showLowpassBtn = tk.Button(self, text='Show after box lowpass mask', command=self.showBoxLowpass)
        self.showLowpassBtn.place(x=280, y=30)
        # highpass
        self.laplacianBtn = tk.Button(self, text='Laplacian of Gaussian operators', command=self.laplacianGuassian)
        self.laplacianBtn.place(x=280, y=210)
        self.multiBtn = tk.Button(self, text='Sobel operators', command=self.sobel)
        self.multiBtn.place(x=280, y=390)
        # order
        self.orderType = tk.StringVar()
        self.orderType.set('median')
        self.orderCombobox = ttk.Combobox(self, values=['median','max','min'],textvariable= self.orderType, state="readonly")
        self.orderCombobox.place(x=400, y=555)
        self.nValue2 = tk.IntVar()
        self.nValue2.set(3)
        self.nCombobox2 = ttk.Combobox(self, values=[3,5,7,9,11,13,15,17,19,21] ,textvariable = self.nValue2, state="readonly")
        self.nCombobox2.place(x=565, y=555)
        self.orderBtn = tk.Button(self, text='order-statistic filter', command=self.order)
        self.orderBtn.place(x=280, y=550)
        

        # show full np matrix
        np.set_printoptions(threshold=np.inf)

    def readJPG(self):
        self.filename = filedialog.askopenfilename()
        try:
            self.image = Image.open(self.filename).convert('L')
            self.image = self.image.resize((120,120), Image.ANTIALIAS) # remember to return
            # to tkinter
            imagetk = ImageTk.PhotoImage(self.image)
            self.imageLabel = tk.Label(self, image= imagetk, width=120, height=120)
            self.imageLabel.image = imagetk
            self.imageLabel.place(x=100, y=60)

        except:
            messagebox.showwarning(title="Warning",message="Please select a JPG file.")


    def showGuassianLowpass(self):
        # create new filter and standardization
        x, y = np.mgrid[0-((self.nValue.get()-1)/2):(self.nValue.get()+1)/2, 0-((self.nValue.get()-1)/2):(self.nValue.get()+1)/2]
        self.lowpassGaussianKernel = np.exp(-(x**2+y**2))
        self.lowpassGaussianKernel = self.lowpassGaussianKernel / self.lowpassGaussianKernel.sum()
        self.showAfterMask(self.lowpassGaussianKernel, int(self.nValue.get()-1)/2)

    def showBoxLowpass(self):
        # create new filter
        lowpassKernel = np.ones((self.nValue.get(),self.nValue.get()))
        lowpassKernel = lowpassKernel / lowpassKernel.sum()
        self.showAfterMask(lowpassKernel, int(self.nValue.get()-1)/2)

    def showAfterMask(self, kernel, padding):
        self.imageNew1 = self.convolve2D(self.image, kernel, padding=int(padding))    
        self.image1 = Image.fromarray(self.imageNew1)
        imagetk = ImageTk.PhotoImage(self.image1)
        self.imageLB1 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageLB1.image = imagetk
        self.imageLB1.place(x = 280, y=60)
        
    
    def convolve2D(self, image, kernel, padding=0, strides=1):
        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        # xImgShape, yImgShape = image.size
        xImgShape =120
        yImgShape =120        

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        self.output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((xImgShape + int(padding)*2, yImgShape + int(padding)*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
            # print(imagePadded)
        else:
            imagePadded = image

        # Iterate through image
        for y in range(yImgShape):
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(xImgShape):
                    # Go to next row once kernel is out of bound
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            self.output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return self.output

    def laplacianGuassian(self):
        highpassGaussianKernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # [[0,-1,0],[-1,4,-1],[0,-1,0]]
        highpassGaussianKernel = np.flipud(np.fliplr(highpassGaussianKernel))# Cross Correlation turn 180o
        self.imageNew2 = self.convolve2D(self.imageNew1, highpassGaussianKernel, padding=int(self.nValue.get()-1)/2)
        self.image2 = Image.fromarray(self.imageNew2)
        imagetk = ImageTk.PhotoImage(self.image2)
        self.imageLB2 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageLB2.image = imagetk
        self.imageLB2.place(x = 280, y=240)

    def sobel(self):
        SobelKernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) 
        self.imageSobel1 = Image.open('Image 3-1.jpg').convert('L')
        self.imageSobel1 = self.imageSobel1.resize((120,120), Image.ANTIALIAS)
        self.imageSobel1 = self.convolve2D(self.imageSobel1, SobelKernel, padding=1)
        self.imageSobel1 = Image.fromarray(self.imageSobel1)
        imagetk = ImageTk.PhotoImage(self.imageSobel1)
        self.imageSobelLB1 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageSobelLB1.image = imagetk
        self.imageSobelLB1.place(x = 280, y=420)

        self.imageSobel2 = Image.open('Image 3-2.jpg').convert('L')
        self.imageSobel2 = self.imageSobel2.resize((120,120), Image.ANTIALIAS)
        self.imageSobel2 = self.convolve2D(self.imageSobel2, SobelKernel, padding=1)
        self.imageSobel2 = Image.fromarray(self.imageSobel2)
        imagetk = ImageTk.PhotoImage(self.imageSobel2)
        self.imageSobelLB2 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageSobelLB2.image = imagetk
        self.imageSobelLB2.place(x = 400, y=420)

        self.imageSobel3 = Image.open('Image 3-3.jpg').convert('L')
        self.imageSobel3 = self.imageSobel3.resize((120,120), Image.ANTIALIAS)
        self.imageSobel3 = self.convolve2D(self.imageSobel3, SobelKernel, padding=1)
        self.imageSobel3 = Image.fromarray(self.imageSobel3)
        imagetk = ImageTk.PhotoImage(self.imageSobel3)
        self.imageSobelLB3 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageSobelLB3.image = imagetk
        self.imageSobelLB3.place(x = 520, y=420)

        self.imageSobel4 = Image.open('Image 3-4.jpg').convert('L')
        self.imageSobel4 = self.imageSobel4.resize((120,120), Image.ANTIALIAS)
        self.imageSobel4 = self.convolve2D(self.imageSobel4, SobelKernel, padding=1)
        self.imageSobel4 = Image.fromarray(self.imageSobel4)
        imagetk = ImageTk.PhotoImage(self.imageSobel4)
        self.imageSobelLB4 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageSobelLB4.image = imagetk
        self.imageSobelLB4.place(x = 640, y=420)

    def order(self, padding=0, strides=1):
        # Initally
        self.imageOrder = Image.open(self.filename).convert('L')
        self.imageOrder = self.imageOrder.resize((120,120), Image.ANTIALIAS)
        padding = int(self.nValue2.get()-1)/2
        # Gather Shapes of Kernel + Image + Padding
        xKernShape = self.nValue2.get()
        yKernShape = self.nValue2.get()
        # xImgShape, yImgShape = image.size
        xImgShape =120
        yImgShape =120
        # get the type of filter        

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((xImgShape + int(padding)*2, yImgShape + int(padding)*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = self.imageOrder
            # print(imagePadded)
        else:
            imagePadded = self.imageOrder

        # Iterate through image
        for y in range(yImgShape):
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(xImgShape):
                    # Go to next row once kernel is out of bound
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            matrix = np.array(imagePadded[x: x + xKernShape, y: y + yKernShape])
                            if(self.orderType.get()=='median'):
                                output[x, y] = np.median(matrix)
                            if(self.orderType.get()=='max'):
                                output[x, y] = np.max(matrix)
                            elif(self.orderType.get()=='min'):
                                output[x, y] = np.min(matrix)
                    except:
                        break
        # print(output)
        self.imageOrder = Image.fromarray(output)
        imagetk = ImageTk.PhotoImage(self.imageOrder)
        self.imageOrderLB4 = tk.Label(self, image= imagetk, width=120, height=120)
        self.imageOrderLB4.image = imagetk
        self.imageOrderLB4.place(x = 280, y=575)


view = grayReader()
view.mainloop()
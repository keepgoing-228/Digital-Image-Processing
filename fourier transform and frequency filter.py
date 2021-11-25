import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.constants import BOTH, LEFT, X, Y
from PIL import Image, ImageTk
import numpy as np
import cv2




class grayReader(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Gray Reader')
        self.geometry('1100x780+100+10')
        self.resizable(0,0)
        self.config(background="LightCyan")
        self.setup()

    def setup(self):
        # input image
        self.inputBtn = tk.Button(self, text='Input JPG File',command=self.readJPG) # if there is () it will run function first
        self.inputBtn.place(x=10, y=5)
        self.FFTBtn = tk.Button(self, text='FFT',command=self.fourier)
        self.FFTBtn.place(x=200, y=5)
        self.IFFTBtn = tk.Button(self, text='IFFT',command=self.inverseFourier)
        self.IFFTBtn.place(x=600, y=5)
        self.testBtn = tk.Button(self, text='Test Original and processed image',command=self.substract)
        self.testBtn.place(x=800, y=5)
        self.cutoff = tk.Spinbox(self,from_=10,to=150,increment=10,width=10)
        self.cutoff.place(x=450,y=300)
        self.idealPassBtn = tk.Button(self, text='Ideal lowpass and highpass',command=self.ideal)
        self.idealPassBtn.place(x=10, y=270)
        self.butterworthPassBtn = tk.Button(self, text='Butterworth lowpass and highpass',command=self.butterworth)
        self.butterworthPassBtn.place(x=200, y=270)
        self.gaussionPassBtn = tk.Button(self, text='Gaussion lowpass and highpass',command=self.gaussian)
        self.gaussionPassBtn.place(x=430, y=270)
        self.homoBtn=tk.Button(self,width=15,text="Homomorphic",command=self.homomorphic)
        self.homoBtn.place(x=700,y=270)
        self.Rl = tk.Spinbox(self,from_=0,to=1,increment=0.1,width=10)
        self.Rl.place(x=710,y=300)
        self.Rh = tk.Spinbox(self,from_=1,to=5,increment=1,width=10)
        self.Rh.place(x=710,y=320)
        self.Do = tk.Spinbox(self,from_=10,to=50,increment=10,width=10)
        self.Do.place(x=710,y=340)
        self.Blur=tk.Button(self,width=15,text="Blur",command=self.blur)
        self.Blur.place(x=10,y=510)

        # show full np matrix
        np.set_printoptions(threshold=np.inf)

    def readJPG(self):
        self.filename = filedialog.askopenfilename()
        try:
            self.image = cv2.imread(self.filename)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.resize(self.image,(200,200))
            # to tkinter
            self.imageOriginal = Image.fromarray(self.image)
            imagetk = ImageTk.PhotoImage(self.imageOriginal)
            self.imageLabel = tk.Label(self, image= imagetk, width=200, height=200)
            self.imageLabel.image = imagetk
            self.imageLabel.place(x=10, y=60)

        except:
            messagebox.showwarning(title="Warning",message="Please select a JPG file.")
    
    def fourier(self):
        ## FFT
        self.fImageArray = np.fft.fft2(self.imageOriginal)
        self.fImageArrayShift = np.fft.fftshift(self.fImageArray)
        
        # normalization
        Fmax = np.log(1+np.max(np.abs(self.fImageArray)))
        Fmin = np.log(1+np.min(np.abs(self.fImageArray)))
        self.fImageArray = 255*((np.log(1+np.abs(self.fImageArray))-Fmin)/(Fmax-Fmin))
        self.fImageArrayShift = 255*((np.log(1+np.abs(self.fImageArrayShift))-Fmin)/(Fmax-Fmin))

        # show FFT
        self.fImage = Image.fromarray(np.uint8(self.fImageArray))
        imagetk = ImageTk.PhotoImage(self.fImage)
        self.fimageLabel = tk.Label(self, image= imagetk, width=200, height=200)
        self.fimageLabel.image = imagetk
        self.fimageLabel.place(x=210, y=60)

        # show the shifted spectrum 
        self.fImageShift = Image.fromarray(np.uint8(np.real(self.fImageArrayShift)))
        imagetk = ImageTk.PhotoImage(self.fImageShift)
        self.fImageShiftLabel = tk.Label(self, image= imagetk, width=200, height=200)
        self.fImageShiftLabel.image = imagetk
        self.fImageShiftLabel.place(x=410, y=60)
        '''
        https://hicraigchen.medium.com/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82
        # import matplotlib.pyplot as plt
        img_c2 = np.fft.fft2(self.imageOriginal)
        img_c3 = np.fft.fftshift(img_c2)
        img_c4 = np.fft.ifftshift(img_c3)
        img_c5 = np.fft.ifft2(img_c4)
        plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
        plt.subplot(151), plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("Spectrum")
        plt.subplot(152), plt.imshow(np.log(1+np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
        plt.subplot(153), plt.imshow(np.log(1+np.abs(img_c4)), "gray"), plt.title("Decentralized")
        plt.subplot(154), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")

        plt.show()
        '''
    
    def inverseFourier(self):
        # inverse fourier
        self.fImageArray = np.fft.fft2(self.imageOriginal)
        self.fImageArrayShift = np.fft.fftshift(self.fImageArray)
        self.inverseFourierImageArrayShift = np.fft.ifftshift(self.fImageArrayShift)
        self.inverseFourierImageArray = np.fft.ifft2(self.inverseFourierImageArrayShift)
        self.inverseFourierImageArray = np.abs(self.inverseFourierImageArray)

        self.inverseFourierImage = Image.fromarray(np.uint8(np.real(self.inverseFourierImageArray)))
        imagetk = ImageTk.PhotoImage(self.inverseFourierImage)
        self.inverseFourierImageLabel = tk.Label(self, image= imagetk, width=200, height=200)
        self.inverseFourierImageLabel.image = imagetk
        self.inverseFourierImageLabel.place(x=610, y=60)

    def ideal(self):
        # lowpass
        h = np.zeros(self.fImageArrayShift.shape) # filter
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    #20為中心圓大小，遍歷並計算頻域中每一個點與這個圓邊界的距離，小於等於20就保留，超過則濾掉
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2) 
                    return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                if dis <= int(self.cutoff.get()):
                    h[i,j]=1
                else:
                    h[i,j]=0
        imgIdealLowArray = h*self.fImageArrayShift
        imgIdealLowArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imgIdealLowArray)))
        imgIdealLow = Image.fromarray(imgIdealLowArray.astype(np.uint8))
        imgIdealLow = imgIdealLow.resize((200,200),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imgIdealLow)
        source = tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=10,y=300)

        # highpass 
        h = np.zeros(self.fImageArrayShift.shape) #reset filter
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    #20為中心圓大小，遍歷並計算頻域中每一個點與這個圓邊界的距離，大於等於20就保留，小於則濾掉
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                if dis <= int(self.cutoff.get()):
                    h[i,j]=0
                else:
                    h[i,j]=1
        imgIdealHighArray = h*self.fImageArrayShift
        imgIdealHighArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imgIdealHighArray)))
        imgIdealHigh = Image.fromarray(imgIdealHighArray.astype(np.uint8))
        imgIdealHigh = imgIdealHigh.resize((200,200),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imgIdealHigh)
        source = tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=210,y=300)
        
    def butterworth(self):
        # lowpass
        h = np.zeros(self.fImageArrayShift.shape) # filter
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2) 
                    return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                h[i,j] = 1/((1+(dis/int(self.cutoff.get())))**2)
        imgIdealLowArray = h*self.fImageArrayShift
        imgIdealLowArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imgIdealLowArray)))
        imgIdealLow = Image.fromarray(imgIdealLowArray.astype(np.uint8))
        imgIdealLow = imgIdealLow.resize((200,200),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imgIdealLow)
        source = tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=10,y=300)

        # highpass 
        h = np.zeros(self.fImageArrayShift.shape) #reset filter
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                h[i,j] = 1/((1+(int(self.cutoff.get())/(dis+0.000001)))**2)
        imgIdealHighArray = h*self.fImageArrayShift
        imgIdealHighArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imgIdealHighArray)))
        imgIdealHigh = Image.fromarray(imgIdealHighArray.astype(np.uint8))
        imgIdealHigh = imgIdealHigh.resize((200,200),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imgIdealHigh)
        source = tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=210,y=300)

    def gaussian(self):
        # lowpass
        h = np.zeros(self.fImageArrayShift.shape) # filter
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2) 
                    return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                h[i,j] = np.exp(-(dis**2)/(2*(int(self.cutoff.get())**2)))
        imgIdealLowArray = h*self.fImageArrayShift
        imgIdealLowArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imgIdealLowArray)))
        imgIdealLow = Image.fromarray(imgIdealLowArray.astype(np.uint8))
        imgIdealLow = imgIdealLow.resize((200,200),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imgIdealLow)
        source = tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=10,y=300)

        # highpass 
        h = np.zeros(self.fImageArrayShift.shape) #reset filter
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                h[i,j] = 1-np.exp(-(dis**2)/(2*(int(self.cutoff.get())**2)))
        imgIdealHighArray = h*self.fImageArrayShift
        imgIdealHighArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imgIdealHighArray)))
        imgIdealHigh = Image.fromarray(imgIdealHighArray.astype(np.uint8))
        imgIdealHigh = imgIdealHigh.resize((200,200),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imgIdealHigh)
        source = tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=210,y=300)

    def homomorphic(self):
        rl=float(self.Rl.get())
        rh=float(self.Rh.get())
        do=int(self.Do.get())
        h=np.zeros(self.fImageArrayShift.shape)
        for i in range (h.shape[0]):
            for j in range (h.shape[1]):
                def cal_distance(pa,pb):
                        from math import sqrt
                        dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                        return dis
                dis = cal_distance((h.shape[0]/2,h.shape[1]/2),(i,j))
                h[i,j] =(rh-rl)*(1-np.exp(-(dis**2)/((do**2))))+rl
        imageHomoArray=h*self.fImageArrayShift
        imageHomoArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imageHomoArray)))
        imageHomo=Image.fromarray(imageHomoArray.astype(np.uint8))
        imageHomo=imageHomo.resize((200,200), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=imageHomo)
        source=tk.Label(self,image=imagetk,width=200,height=200)
        source.image=imagetk
        source.place(x=810,y=300)
        

    def blur(self):        
        h=self.fImageArray
        import math
        for i in range (h.shape[0]):
            for j in range (h.shape[1]):
                def blurr(u,v):
                    a = 0.1
                    b = 0.1
                    T = 1
                    C = (a*u+b*v)
                    if(C == 0):
                        return 1
                    return (T/C)*math.sin(C)*math.e**(-1j*C) #課本的公式輸入進來
                h[i,j] =blurr(i,j);
        imageBlurArray=h*self.fImageArrayShift
        imageBlurArray = np.abs(np.fft.ifft2(np.fft.ifftshift(imageBlurArray)))
        bx = np.max(imageBlurArray)
        bxi = np.min(imageBlurArray)
        for i in range(200):
            for j in range(200):
                imageBlurArray[i][j] = 255*(imageBlurArray[i][j]-bxi)/(bx-bxi)
        img1=Image.fromarray(np.uint8(imageBlurArray))
        img1=img1.resize((200,200), Image.ANTIALIAS)
        imgtk1 = ImageTk.PhotoImage(image=img1)
        source=tk.Label(self,image=imgtk1,width=200,height=200)
        source.image=imgtk1
        source.place(x=10,y=540)

        # inverse filter
        imageBlurArray=h*self.fImageArrayShift
        self.imgback=imageBlurArray/h 
        self.imgback = np.abs(np.fft.ifft2(np.fft.ifftshift(self.imgback)))
        img2=Image.fromarray(self.imgback.astype(np.uint8))
        img2=img2.resize((200,200), Image.ANTIALIAS)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        source=tk.Label(self,image=imgtk2,width=200,height=200)
        source.image=imgtk2
        source.place(x=210,y=540)

        # wiener filter
        self.imgwie=imageBlurArray*(abs(h)*abs(h)/(h*(abs(h)*abs(h)))) 
        self.imgwie=np.abs(np.fft.ifft2(np.fft.ifftshift(self.imgwie)))
        wx = np.max(self.imgwie)
        wxi = np.min(self.imgwie)
        for i in range(200):
            for j in range(200):
                self.imgwie[i][j] = 255*(self.imgwie[i][j]-wxi)/(wx-wxi)
        img3=Image.fromarray(np.uint8(self.imgwie))
        img3=img3.resize((200,200), Image.ANTIALIAS)
        imgtk3 = ImageTk.PhotoImage(image=img3)
        source=tk.Label(self,image=imgtk3,width=200,height=200)
        source.image=imgtk3
        source.place(x=410,y=540)


        def add_gaussian_noise(image):
            mean = 0
            var = 20
            sigma = var ** 0.5
            row,col= image.shape[0],image.shape[1]
            gauss = np.random.normal(mean,sigma,(row,col))
            gauss = gauss.reshape(row,col)
            noisy = image + gauss
            return noisy
        imggosn=add_gaussian_noise(self.fImageArray)
        f1 = np.fft.fft2(imggosn)
        imggosn = np.fft.fftshift(f1)
        imggosiver=imggosn*h/h  # inverse filter
        imggosiver = np.abs(np.fft.ifft2(np.fft.ifftshift(imggosiver)))
        img4=Image.fromarray(imggosiver.astype(np.uint8))
        img4=img4.resize((200,200), Image.ANTIALIAS)
        imgtk4 = ImageTk.PhotoImage(image=img4)
        source=tk.Label(self,image=imgtk4,width=200,height=200)
        source.image=imgtk4
        source.place(x=610,y=540)

        imggosn=imggosn*h
        imggoswie=imggosn*(abs(h)*abs(h)/(h*(abs(h)*abs(h)+10)))  #wiener filter
        imggoswie=np.abs(np.fft.ifft2(np.fft.ifftshift(imggoswie)))
        yx = np.max(imggoswie)
        yxi = np.min(imggoswie)
        for i in range(200):
            for j in range(200):
                imggoswie[i][j] = 255*(imggoswie[i][j]-yxi)/(yx-yxi)
        img5=Image.fromarray(np.uint8(imggoswie))
        img5=img5.resize((200,200), Image.ANTIALIAS)
        imgtk5 = ImageTk.PhotoImage(image=img5)
        source=tk.Label(self,image=imgtk5,width=200,height=200)
        source.image=imgtk5
        source.place(x=810,y=540)


    def substract(self):
        self.substractMatrix = np.zeros((200,200))        
        for y in range(200):
            for x in range(200):
                self.substractMatrix[y][x] = self.image[y][x] - np.uint8(np.real(self.inverseFourierImageArray[y][x]))
                # inversed image substract wiener filter image
                # self.substractMatrix[y][x] = self.imgwie[y][x] - self.imgback[y][x] 
                # self.substractMatrix[y][x] = self.imgback[y][x] - self.imgwie[y][x]
                if self.substractMatrix[y][x] < 0:
                    self.substractMatrix[y][x] = 255
        self.substractImage = Image.fromarray(np.uint8(self.substractMatrix))
        self.substractImage = self.substractImage.resize((200,200), Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(self.substractImage)
        self.substractLabel=tk.Label(self, image=imagetk, width=200, height=200)
        self.substractLabel.image=imagetk
        self.substractLabel.place(x=810,y=60)






view = grayReader()
view.mainloop()

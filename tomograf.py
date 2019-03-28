import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from skimage import data
import dicom

'''Podajemy wspolrzedne dwoch punktow, algorytm oblicza dlugosc odcinka, ktory tworza.
Dziala tylko na liczbach calkowitych
image-obraz wejsciowy
xa,ya,xb,yb-wspolrzedne koncow odcinka
w,h-srodek obrazu
inv=True-stosujemy algorytm dla odwrotnej transformaty Radona
'''


def bresenham(image,xa,xb,ya,yb,w,h,inv=False,col=1):
    points=[]
    if xa<xb:
        x=1
        dx=xb-xa;
    else:
        x=-1
        dx=xa-xb

    if ya<yb:
        y=1
        dy=yb-ya
    else:
        y=-1
        dy=ya-yb  
        
    if xa>=0 and xa<w and ya>=0 and ya<h:
        if inv==False:
            color = image[h-1-ya][xa]
            if color > 0: points.append(color)
        else:
            image[h-1-ya][xa] += col
    if dx>dy:
        a=(dy-dx)*2
        b=dy*2
        d=b-dx
        while xa!=xb:
            if d>=0:
                xa+=x
                ya+=y
                d+=a
            else:
                d+=b
                xa+=x
            if xa>=0 and xa<w and ya>=0 and ya<h:
                if inv==False:
                    color = image[h-1-ya][xa]
                    if color > 0: points.append(color)
                else:
                    image[h-1-ya][xa] += col
    else:
        a=(dx-dy)*2
        b=dx*2
        d=b-dy
        while ya!=yb:
            if d>=0:
                xa+=x
                ya+=y
                d+=a
            else:
                d+=b
                ya+=y
            if xa>=0 and xa<w and ya>=0 and ya<h:
                if inv==False:
                    color = image[h-1-ya][xa]
                    if color > 0: points.append(color)
                else:
                    image[h-1-ya][xa] += col
    if inv==False:
        return points
    else:
        return image

#rysuje obraz po transformacie
def drawRadon(img,z):
    pom=img/max(img.flatten())
    cv2.imshow('sinogram',pom)
    cv2.waitKey(1000)
    
def drawInvRadon(img,z):
    img -= min(img.flatten())
    pom =img/ max(img.flatten())
    cv2.imshow('obraz końcowy',pom)
    cv2.waitKey(1000)
    
'''Petla wyznacza kolejne punkty, dla ktorych stosowany jest algorytm  bresenhama.
Zastosowano model emiterow/detektorow rownolegly'''
def radonRepeat(inputImg,outputImg,w,h,nsteps,alfa,ndetectors=360,l=360,inv=False):
    #odleglosc miedzy dwoma detektorami
    detectorsDistance=l/(ndetectors-1)
    z=1
    r=math.sqrt(w*w+h*h) #promien okregu opisanego
    alfa=alfa*math.pi/180 #kat o ktory sie przesuwamy w radianach
    
    for i in range(nsteps):
        angle=alfa*i
        for j in range(0,ndetectors):
            d=l/2-detectorsDistance*j
            dsign=1
            if d<0:
                dsign=-1
                d=abs(d)
            
            sinus=abs(math.sin(angle))
            cosinus=abs(math.cos(angle))
            tangens=abs(math.tan(angle))
            a=math.sqrt(r*r-d*d)
            b=0
            if sinus<0.0001:
                ya=int(a+h)
                yb=int(h-a)
                if dsign==1:
                    xa=int(w+d)
                    xb=int(w+d)
                else:
                    xa=int(w-d)
                    xb=int(w-d)
            else:
                b=d/tangens
                y2=d/sinus
                
                if angle<=1.57:
                    if dsign==1:
                        xa=int(w+(a-b)*sinus)
                        xb=int(w-(a+b)*sinus)
                        y1=int((a-b)*cosinus)
                        ya=int(y1+y2+h)
                        y1=int((a+b)*cosinus)
                        yb=int(h-(y1-y2))
                    else:
                        xa=int(w+(a+b)*sinus)
                        xb=int(w-(a-b)*sinus)
                        y1=int((a+b)*cosinus)
                        ya=int(y1-y2+h)
                        y1=int((a-b)*cosinus)
                        yb=int(h-y1-y2)
                else:
                    if dsign==1:
                        xa=int(w+(a+b)*sinus)
                        xb=int(w-(a-b)*sinus)
                        y1=int((a+b)*cosinus)
                        ya=int(h-(y1-y2))
                        y1=int((a-b)*cosinus)
                        yb=int(h+y1+y2)
                    else:
                        xa=int(w+(a-b)*sinus)
                        xb=int(w-(a+b)*sinus)
                        y1=int((a-b)*cosinus)
                        ya=int(h-y1-y2)
                        y1=int((a+b)*cosinus)
                        yb=int(h+y1-y2)
            
            if inv==False:
                points=bresenham(inputImg,xa,xb,ya,yb,inputImg.shape[1],inputImg.shape[0])
                if len(points)>0:
                   outputImg[j][i]=sum(points) 
            else:
                color=inputImg[j,i]
                outputImg=bresenham(outputImg,xa,xb,ya,yb,inputImg.shape[0],inputImg.shape[0],True,color)
        if i%10==0 and inv==False:
            drawRadon(outputImg,z)
            z+=1
        if i%10==0 and inv==True:
            drawInvRadon(outputImg,z+20)           
            z+=1
    if inv==False:
        drawRadon(outputImg,z)
    else:
       drawInvRadon(outputImg,z+20)           
    return outputImg


'''
model rownolegly
image-obraz wejsciowy
alfa-kat, o ktory obracamy emiter/detektor (w stopniach)
ndetectors-liczba detektorow
l-rozpietosc detektorow
sinogram-obraz wyjsciowy
 '''
def radon(image,alfa=1,ndetectors=360,l=360): 
    #w,h- srodek zdjecia
    w=image.shape[1]//2
    h=image.shape[0]//2 
    nsteps=180//alfa
    sinogram = np.zeros((ndetectors,nsteps+1))

    sinogram=radonRepeat(image,sinogram,w,h,nsteps,alfa,ndetectors,l)
    
    #normalizacja
    sinogram /= max(sinogram.flatten())
    return sinogram

def inverseRadon(sinogram,alfa=1,ndetectors=360,l=360):
    h=sinogram.shape[0]//2
    w=h
    nsteps=180//alfa+1
    image = np.zeros((sinogram.shape[0],sinogram.shape[0]))

    image=radonRepeat(sinogram,image,w,h,nsteps,alfa,ndetectors,l,True)

    #normalizacja
    image -= min(image.flatten())
    image /= max(image.flatten())

    return image


image = cv2.imread("Kwadraty2.jpg", 0)

#dodaje czarne piksele tworzac kwadratowe zdjecie
test=np.zeros(len(image[0]))
if image.shape[1]>image.shape[0]:
    test=np.zeros((image.shape[1],image.shape[1]))
    for i in range(len(image)):
        test[i]=image[i]
else:
    test=np.zeros((image.shape[0],image.shape[0]))
    for i in range(len(image)):
        for j in range(len(image[0])):
            test[i][j]=image[i][j]
    
n=int(len(test)*0.4) #liczba detektorow
l=int(len(test)*0.7) #rozpietosc detektorow    
#wykonanie radona
radonSin=radon(test,ndetectors=n,l=l)
#l-czyli rozpietosc zmienia sie proporcjonalnie do zmiany wielkosci obrazu 
l=n*l/len(test)
radonInv=inverseRadon(radonSin,ndetectors=n,l=l)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''plt.subplot(3, 1, 1)
plt.imshow(test, cmap='gray', interpolation=None)
plt.subplot(3, 1, 2)
plt.imshow(radonSin, cmap='gray', extent=[0,180,len(radonSin),0], interpolation=None)
plt.subplot(3, 1, 3)
plt.imshow(radonInv, cmap='gray',  interpolation=None)
'''

def loadDICOM(filename):
    file = dicom.read_file(filename)

    name = str(file.PatientName)
    name = name.split('^')
    lastName = name[0]
    firstName = name[1]
    id = str(file.PatientID)
    birthday = str(file.PatientBirthDate)
    birthday = birthday[0:4] + "-" + birthday[4:6] + "-" + birthday[6:8]
    gender = str(file.PatientSex)
    date = str(file.StudyDate)  # Study date
    time = str(file.StudyTime)   # Study time
    date = date[0:4]+"-"+date[4:6]+"-"+date[6:8]
    time = time[0:2]+":"+time[2:4]+":"+time[4:6]
    image = file.pixel_array

    return image

    #return lastName, firstName, id,birthday, gender, date, time, image

def loadNormalImage(filename):
    image = cv2.imread(filename, 0)
    return image

def checkImage(image):
    test = np.zeros(len(image[0]))
    if image.shape[1] > image.shape[0]:
        test = np.zeros((image.shape[1], image.shape[1]))
        for i in range(len(image)):
            test[i] = image[i]
    else:
        test = np.zeros((image.shape[0], image.shape[0]))
        for i in range(len(image)):
            for j in range(len(image[0])):
                test[i][j] = image[i][j]
    return test
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import pandas as pd


def gabor(img:np.ndarray):
  buff=np.copy(img)
  gamma=1.2
  sigma = 12
  ksize=30
  output=[]
  for theta in range(8):
    theta = theta/8. * np.pi
    for lamda in [1/0.25,1/0.18,1/0.13,1/0.09,1/0.06]:
      kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,0,ktype=cv2.CV_32F)
      fimg=cv2.filter2D(buff,cv2.CV_8UC3,kernel)
      output.append(np.copy(fimg))
  output=np.array(output)
  return output

def law_texture(img:np.ndarray):
  buff=np.copy(img)
  #5 vectors from which the masks are created:
  L5=np.array([1,4,6,4,1])#(Level)
  E5=np.array([-1,-2,0,2,1])#(Edge)
  S5=np.array([-1,0,2,0,-1])#(Spot)
  R5=np.array([1,-4,6,-4,1])#(Ripple)
  vectors=[L5,E5,S5,R5]
  masks=[]
  for x in vectors:
    for y in vectors:
      masks.append(np.outer(x.T,y))
  k=np.copy(buff[8:buff.shape[0]-7,8:buff.shape[1]-7])
  z=np.zeros((buff.shape[0]-15,buff.shape[1]-15))
  #removing ilumination
  for x in range(0,15):
        for y in range(0,15):
          z+=(np.copy(buff[x:buff.shape[0]-15+x,y:buff.shape[1]-15+y])/(15**2))
  k=k-z
  k[k<0]=0
  buff=k
  plt.imshow(buff, cmap='gray')
  mas=[]
  for m in masks:
    suma=np.zeros((buff.shape[0]-5,buff.shape[1]-5))
    for x in range(0,5):
        for y in range(0,5):
          suma+=(np.copy(buff[y:buff.shape[0]-5+y,x:buff.shape[1]-5+x])*np.copy(m[x,y]))
    suma[suma<0]=0
    suma[suma>255]=255
    mas.append(np.copy(suma))
  final=[(mas[1]+mas[4])/2,(mas[3]+mas[12])/2,(mas[6]+mas[9])/2,mas[10],mas[15],(mas[2]+mas[8])/2,mas[5],(mas[7]+mas[13])/2,(mas[11]+mas[14])/2]
  return final

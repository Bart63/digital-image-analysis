import numpy as np


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

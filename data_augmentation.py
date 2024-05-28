import os
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import random
#import xlsxwriter
import json
import csv

#Funzione di brightness per data augmentation
def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img

#Funzione di rotation per data augmentation
def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv.warpAffine(img, M, (w, h))
    return img
    

path= "/home/leo/Scrivania/AV_Project/trainingset_av_2021_22/utkface/"

#Creazione dizionario {"nome_img": label}
cs=pd.read_csv("/home/leo/Scrivania/AV_Project/labels/labels.csv")
df=pd.DataFrame(cs)
l=["10_0_0_20170110220557169.jpg.chip.jpg","0","0.1","0.2"]
df=df.rename(columns={l[0]: 0, l[1]:1,l[2]:2,l[3]:3})
keys=list(df[0])
dict={}
for i in range(len(keys)):
    dict[keys[i]]= [str(df[1][i]),str(df[2][i]),str(df[3][i])]
dict[l[0]]=["0","0","0"]
dict2={}

#Applicazione di data augmentation
j=0
labels=[]
for i in dict:

    #Data augmentation per tutti i campioni che abbiano o barba o baffi o occhiali
    if dict[i]!= ["0","0","0"]:
        img = cv.imread(path+i)

        #doing britghness
        dst= brightness(img, 0.5, 3)
        imgname= "brightness_"+str(j)+".jpg"
        name=path+imgname
        cv.imwrite(name, dst)
        labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
        dict2[imgname]=[dict[i][0],dict[i][1],dict[i][2]]

        #doing rotation
        dst = rotation(img, 30)
        imgname="rotation_"+str(j)+".jpg"
        name=path+imgname
        cv.imwrite(name, dst)
        labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
        dict2[imgname]=[dict[i][0],dict[i][1],dict[i][2]]
        
        #doing blurring
        img = cv.imread(path+i)
        imgname= "blurred_"+str(j)+".jpg"
        name=path+imgname
        blur = cv.blur(img,(5,5))
        labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
        dict2[imgname]=[dict[i][0],dict[i][1],dict[i][2]]
        cv.imwrite(name, blur)
        
        #Augmentation aggiuntiva per classi di campioni di numero inferiore
        img = cv.imread(path+i)
        if dict[i]==["0","1","0"] or dict[i]==["1","0","1"]:
            #doing rotation 60°
            dst = rotation(img, 60)
            imgname="rotation60_"+str(j)+".jpg"
            name=path+imgname
            cv.imwrite(name, dst)
            labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
            dict2[imgname]=[str(dict[i][0]),str(dict[i][1]),str(dict[i][2])]
            #doing rotation 90°
            dst = rotation(img, 90)
            imgname="rotation90_"+str(j)+".jpg"
            name=path+imgname
            cv.imwrite(name, dst)
            labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
            dict2[imgname]=[str(dict[i][0]),str(dict[i][1]),str(dict[i][2])]

        elif dict[i]==["1","0","0"]:
            #doing rotation 60°
            dst = rotation(img, 60)
            imgname="rotation60_"+str(j)+".jpg"
            name=path+imgname
            cv.imwrite(name, dst)
            labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
            dict2[imgname]=[str(dict[i][0]),str(dict[i][1]),str(dict[i][2])]

        elif dict[i]==["0","1","1"]:
            #doing rotation 60°
            dst = rotation(img, 60)
            imgname="rotation60_"+str(j)+".jpg"
            name=path+imgname
            cv.imwrite(name, dst)
            labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
            dict2[imgname]=[str(dict[i][0]),str(dict[i][1]),str(dict[i][2])]
            #doing rotation 90°
            dst = rotation(img, 90)
            imgname="rotation90_"+str(j)+".jpg"
            name=path+imgname
            cv.imwrite(name, dst)
            labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
            dict2[imgname]=[str(dict[i][0]),str(dict[i][1]),str(dict[i][2])]
            #doing rotation 180°
            dst = rotation(img, 180)
            imgname="rotation180_"+str(j)+".jpg"
            name=path+imgname
            cv.imwrite(name, dst)
            labels.append(imgname+","+str(dict[i][0])+","+str(dict[i][1])+","+str(dict[i][2]))
            dict2[imgname]=[str(dict[i][0]),str(dict[i][1]),str(dict[i][2])]
        j+=1

#saving json label
dict3={**dict, **dict2}
with open("/home/leo/Scrivania/AV_Project/labels/label.json", "w") as outfile:
    json.dump(dict3, outfile)


#saving labels
'''
workbook = xlsxwriter.Workbook("C:/Users/user/Desktop/ProgettoArtificial/newlabel.csv")
worksheet = workbook.add_worksheet()
row = 0
column = 0
for label in labels :
    worksheet.write(row, column, label)
    row += 1
workbook.close()

'''
with open('/home/leo/Scrivania/AV_Project/labels/newlabel.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    for label in labels:
        writer.writerow([label])


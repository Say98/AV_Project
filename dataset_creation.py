import numpy as np
import json
import cv2 as cv
import h5py
from random import shuffle
from shutil import copyfile
import xlwt

path="/home/leo/Scrivania/AV_Project/trainingset_av_2021_22/utkface/"
out="/home/leo/Scrivania/AV_Project/trainingset_av_2021_22/validation_set/"

wb = xlwt.Workbook()

# open the first sheet
w_sheet = wb.add_sheet('0',cell_overwrite_ok=True)

#CREATE H5 DATASET
with open('/home/leo/Scrivania/AV_Project/labels/label.json') as json_file:
    labels = json.load(json_file)


keys = list(labels.keys())
shuffle(keys)
num_train = int(len(keys)*0.8)
train = keys[:num_train]
validation = keys[num_train:]

X_t_list=[]
y_t_list=[]

for img_name in train:
    image_name = path + img_name
    img = cv.imread(image_name)
    img_arr= np.asarray(img)
    X_t_list.append(img_arr)
    y_t_list.append(np.asarray(labels[img_name]).astype(int))
X_train = np.array(X_t_list)
y_train = np.array(y_t_list)

X_v_list=[]
y_v_list=[]
i=0
for img_name in validation:
    copyfile(path+img_name, out+img_name)
    w_sheet.write(i,0,img_name)

    image_name = path + img_name
    img = cv.imread(image_name)
    img_arr= np.asarray(img)
    X_v_list.append(img_arr)
    y_v_list.append(np.asarray(labels[img_name]).astype(int))

    w_sheet.write(i,1,labels[img_name][0])
    w_sheet.write(i,2,labels[img_name][1])
    w_sheet.write(i,3,labels[img_name][2])
    i+=1

X_val=np.array(X_v_list)
y_val=np.array(y_v_list)

wb.save('/home/leo/Scrivania/AV_Project/val_labels.xls')

f = h5py.File("/home/leo/Scrivania/AV_Project/dataset_final.hdf5", mode='w')

f.create_dataset("X_train", data=X_train)
f.create_dataset("y_train", data=y_train)
f.create_dataset("X_val", data=X_val)  
f.create_dataset("y_val", data =y_val)

f.close

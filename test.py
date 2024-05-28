import os, sys, cv2, csv, argparse, random
import numpy as np
from keras.models import load_model
from tqdm import tqdm

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='val_labels.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='validation_set/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='vgg_results.csv', help="CSV file of the results")
    parser.add_argument("--model", type=str, default='vgg_model_2022_01_07_13_53.h5', help="Model file .h5")
    args = parser.parse_args()
    return args


def F1_score(y_true, y_pred):
    '''Define f1 measurement'''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return 2*((precision * recall) / (precision + recall + K.epsilon()))

args = init_parameter()
model = load_model(args.model,custom_objects={'F1_score': F1_score})


# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    b_dict = {}
    m_dict = {}
    g_dict = {}
    for row in gt:
        b_dict.update({row[0]: int(round(float(row[1])))})
        m_dict.update({row[0]: int(round(float(row[2])))})
        g_dict.update({row[0]: int(round(float(row[3])))})
        gt_num += 1

# Opening CSV results file
k=0
with open(args.results, 'w', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in b_dict.keys():
        k+=1
        img = cv2.imread(args.images+image)
        if img.size == 0:
            print("Error")
        img = cv2.resize(img, (200, 200))
        img = np.reshape(img, (1,200,200,3))
        predictions = model.predict(img)[0]
        predictions = [round(prediction,3) for prediction in predictions]
        
        if predictions[0]>=0.5:
            b=1
        else:
            b=0
        if predictions[1]>=0.5:
            m=1
        else:
            m=0
        if predictions[2]>=0.5:
            g=1
        else:
            g=0

        if (b==1 and m==1) == 1:
            m = 0

        # Writing a row in the CSV file
        writer.writerow([image, b, m, g])
print("Done")
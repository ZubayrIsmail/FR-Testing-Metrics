import time, math, os, data 
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def loadDataATT():
    training_dir = 'Datasets/att_faces/Training'
    testing_dir = 'Datasets/att_faces/Testing'
    imageWidth = 92
    imageHeight = 112    
    
    
    X, Y= data.LoadTrainingData(training_dir, (imageWidth, imageHeight))
    data.TrainingData = X
    data.TrainingLables = Y    
    
    XT, YT, NamesT, _, Paths = data.LoadTestingData(testing_dir, (imageWidth, imageHeight))
    data.TestingData = XT
    data.TestingLables = YT    
    
    #convert from one hot encoding to array of actual ID's must move to data file
    Y = [np.where(target==1)[0][0] for target in Y]
    YT = [np.where(target==1)[0][0] for target in YT]
    
    return X,Y,XT,YT


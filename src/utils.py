# -*- coding: utf-8 -*-

# Pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow libraries
import tensorflow as tf

def cleanTCSdataset(df):
    df_copy = df.copy()
    del df_copy['timestamp']
    del df_copy['Day']
    del df_copy['Time Broken']
    del df_copy['Day_Time']
    del df_copy['Text(Day_Time)']
    del df_copy['Total Seconds']
    del df_copy['Diff Seconds']
    del df_copy['Temperature']
    del df_copy['ThermalComfort']
    del df_copy['TopClothing']
    del df_copy['BottomClothing']
    del df_copy['OuterLayerClothing']
    del df_copy['ActivityDescription']
    del df_copy['Thermal Comfort TA']
    del df_copy['Activity']
    del df_copy['Gsr']
    
    # df_copy.dropna(axis=0, inplace=True)
    df_copy = df_copy.fillna(0) # fill NaN with 0

    df_survey = df_copy[df_copy['class'] == 'SurveyData']
    df_survey.reset_index(inplace=True, drop=True)
    df_band = df_copy[df_copy['class'] == 'BandData']
    df_band.reset_index(inplace=True, drop=True)

    del df_band['class']
    del df_survey['class']    
    # categorical to numbers
    #m = {'Male' : 0, 'Female' : 1}  
    #df_copy['Gender'] = df_copy['Gender'].map(m)

    print("Number of Survey instances: ", len(df_survey), "\n")
    print("Number of Band instances: ", len(df_band), "\n")

    return df_band, df_survey

def checkGPU():
    if tf.test.is_gpu_available():
        print("Current GPU: {}".format(tf.test.gpu_device_name()))
        gpu = tf.test.gpu_device_name()
    else:
        print("No GPU will be used")
        gpu = None
    return gpu

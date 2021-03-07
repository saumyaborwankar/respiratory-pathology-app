
import os
import streamlit as st
import io
import librosa
import pickle 
import numpy as np
import keras
model = keras.models.load_model('model/model_kfold.h5')
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# from moviepy.editor import VideoFileClip, concatenate_videoclips



def respiratory_pathology_detect(filename):
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=39).T,axis=0) 
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=39,fmax=8000).T,axis=0)
    #chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    #chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=X, sr=sample_rate,n_chroma=39,bins_per_octave=39).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_cens)),(39,3))
    #now making labels p225, p235 etc
    # file = (file[0:3])  
    #arr = mfccs, file
    pkl_file = open('model/Departure_encoder.pkl', 'rb')
    pathology = pickle.load(pkl_file) 
    pkl_file.close()
    X_new=features
    X_new=np.reshape(X_new,(1,39,3,1))
    a=model.predict_classes(X_new)
    print(a)
    result=pathology.inverse_transform(a)
    print(result)
    return result[0]

    


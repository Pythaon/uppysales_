# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:14:48 2021

@author: Utilisateur
"""

import pandas as pd
import numpy as np
import datetime 
import time
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
from PIL import Image
import requests
import io
from io import StringIO
import os

##------- ALL PAGE 
st.set_page_config(page_title="UpPySales App",page_icon="🎯",layout="wide",initial_sidebar_state="expanded")

page = st.sidebar.radio(label="Menu", options = ['1️⃣ Présentation',  '2️⃣ Segmentation visiteurs', '3️⃣ Clustering'])

### LOGO
@st.cache
def img():
    urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/uppysales_s.png"
    image = Image.open(requests.get(urllogo, stream=True).raw)
    return image
    
image = img()
    
st.image(image, width=None)

st.title("""**Analyse de l'activité de e-commerce**""")

##------- IMPORT DES DATASETS
@st.cache
def load_data1():
    url_events="http://spowls.net:449/projet/datasets/events.csv"
    s_events=requests.get(url_events).content 
    events=pd.read_csv(io.StringIO(s_events.decode('utf-8')))
    return events

### Df_all sample random 30% 
@st.cache
def load_data2():
    url_df_all="http://spowls.net:449/projet/datasets/df_all_sample30.csv"
    s_df_all=requests.get(url_df_all).content 
    df_all=pd.read_csv(io.StringIO(s_df_all.decode('utf-8')))
    return df_all
  
events = load_data1()
df_all = load_data2()



if page =='Clustering':
    
    st.title ("Clustering")
    
    
    st.markdown("""
                Nous allons tester les modèles suivants:""")
                
    models = ['Kmeans', 'Clustering Mixte kmeans & ACH']
              
    choix_modele = st.radio("", options=models)
    
    if choix_modele ==models[0]:
    
        @st.cache
        def img():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.PNG"
            image = Image.open(requests.get(urllogo, stream=True).raw)
            return image

        image = img()
    

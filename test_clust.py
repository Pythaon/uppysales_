# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:52:38 2021

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

    
    
st.title ("Clustering")


st.markdown("""
            Nous allons tester les modèles suivants:
                
                
                """)
            
models = ['Kmeans', 'Clustering Mixte kmeans & ACH']
          
choix_modele = st.radio("", options=models)



if choix_modele ==models[0]:
    
    #@st.cache
    #def img():
     #   urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.png"
      #  image = Image.open(requests.get(urllogo, stream=True).raw)
       # return image
    
    #image = img()

    @st.cache
    def img():
        urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.PNG"
        image = Image.open(requests.get(urllogo, stream=True).raw)
        return image
    
    image = img()
    
    
    st.write("""
             Le coude n'est pas très franc mais apparait autour du nombre de cluster = 4
             """)

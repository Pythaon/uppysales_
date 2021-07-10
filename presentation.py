# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:18:48 2021
@author: Céline & Thao
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

page = st.sidebar.radio(label="Menu", options = ['Présentation',  'Segmentation visiteurs', 
                                  'Clustering'])
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

##------- PAGE PRÉSENTATION
if page == 'Présentation':
    ### LOGO
    urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/c277f4b10bcee7a1fd364ef3bab9497d22900863/uppysales.png"
    image = Image.open(requests.get(urllogo, stream=True).raw)
    
    st.image(image)
    
             
    st.header("""**Analyse de l'activité de e-commerce**""")         
             
    st.write(""" 
             
            Le jeu de données est un journal d'activité, il contient donc des timestamps, 
            des actions de visiteurs sur le site pendant 139 jours.
            Nous connaissons l'activité des visiteurs: l’article (itemid) visité, 
            le/les mis au panier, ou acheté(s), le prix de l'item et sa disponibilité.
            La nature des articles est anonymisée (nous ne connaissons pas le contenu du site).
             
             """)
    
    ### -- 
    st.write("""
             
            ##La table principale est la table event:
             
             """)
    st.dataframe(events.head())
    
    st.write(""" 
             
            La table est très étroite avec seulement 5 variables, de plus le jeu de données est
            déséquilibré sur la variable **transactionid** avec moins de **1%** des enregistrements 
            qui concernent des transactions.
             
             """)
   
    plt.figure(figsize=(8,6))         
    fig, ax = plt.subplots()
    sns.countplot(df_all['event'], ax=ax)
    st.pyplot(fig)
    
    ### -- 
    
    st.write(""" 
             
           Afin de donner un peu de largeur au dataset, nous avons ajouté des variables calculées 
           (ex :durées entre les évènements de vue, mise au panier, achat, prix, disponibilité des items).
             
             """)
   
  
    st.write("""
             
            ##La table ci-dessous est un sample random de 30% du dataset principal retravaillé 
             
             """)
    
    st.dataframe(df_all.head())
    
    
   

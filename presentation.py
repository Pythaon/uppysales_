# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:18:48 2021

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

page = st.sidebar.radio(label="Menu", options = ['Présentation',  'Segmentation visiteurs', 
                                  'Clustering'])

###TEST
DATE_COLUMN = 'timestamp'
DATA_URL = ('http://spowls.net:449/projet/datasets/item_properties_part1.csv')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

##TEST


@st.cache
def load_data1():
    url_events="http://spowls.net:449/projet/datasets/events.csv"
    s_events=requests.get(url_events).content 
    events=pd.read_csv(io.StringIO(s_events.decode('utf-8')))
    return events

#@st.cache
#def load_data2():
    #url_dfall="http://spowls.net:449/projet/datasets/df_all.csv"
    #s_dfall=requests.get(url_dfall).content
    #df_all=pd.read_csv(io.StringIO(s_dfall.decode('utf-8')))
    #return df_all
  
events = load_data1()
#df_all = load_data2()


if page == 'Présentation':
    ### LOGO
    urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/c277f4b10bcee7a1fd364ef3bab9497d22900863/uppysales.png"
    image = Image.open(requests.get(urllogo, stream=True).raw)
    
    st.image(image)
    
             
    st.header("""Analyse de l'activité de e-commerce""")         
             
    st.write(""" 
             
            Le jeu de données est un journal d'activité, il contient donc des timestamps, des actions des
            visiteurs sur le site pendant 139 jours.
            Nous connaissons l'activité des visiteurs: l’article (itemid) visité, le/les mis au panier, ou
            acheté(s), le prix de l'item et sa disponibilité.
            La nature des articles est anonymisée (nous ne connaissons pas le contenu du site).
             
             La table principale est la table event:
             
             """)
    
    ### -- 
    
    st.dataframe(events.head())
    
    st.write(""" 
             
            La table est très étroite avec seulement 5 variables, de plus le jeu de
             données es déséquilibré sur la variable **transactionid** avec moins de 1% des enregistrements qui concernent des transactions
             
             """)
   
             
    #fig, ax = plt.subplots()
    #sns.countplot(df_all['event'], ax=ax)
    #st.pyplot(fig)
    
    ### -- 
    
    st.write(""" 
             
           Afin de donner un peu de largeur au dataset, nous avons ajouté des variables calculées 
           (ex :durées entre les évènements de vue, mise au panier, achat, prix, disponibilité des items).
             
             """)
    
    #st.dataframe(df_all.head())
    
    
   

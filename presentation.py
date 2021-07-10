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


page = st.sidebar.radio(label="Menu", options = ['Présentation',  'Segmentation visiteurs', 
                                  'Clustering', 'test'])

#@st.cache

#def chargement ():
    #df_all=pd.read_csv(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\Dataset\df_all.csv')
     #return pd.read_csv(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\Dataset\events.csv')
   
#events = chargement()

#events = pd.read_csv(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\Dataset\events.csv')
#df_all=pd.read_csv(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\Dataset\df_all.csv')


#url="http://spowls.net:449/projet/datasets/events.csv"
#s=requests.get(url).content
#events=pd.read_csv(io.StringIO(s.decode('utf-8')))

#url1="http://spowls.net:449/projet/datasets/df_all.csv"
#s1=requests.get(url1).content
#df_all=pd.read_csv(io.StringIO(s1.decode('utf-8')))


if page == 'Présentation':

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
             
    url_events="http://spowls.net:449/projet/datasets/events.csv"
    s_events=requests.get(url_events).content
    
    events=pd.read_csv(io.StringIO(s_events.decode('utf-8')))
    
    st.dataframe(events.head())
    
    st.write(""" 
             
            La table est très étroite avec seulement 5 variables, de plus le jeu de
             données es déséquilibré sur la variable **transactionid** avec moins de 1% des enregistrements qui concernent des transactions
             
             """)
    
    url_df_all="http://spowls.net:449/projet/datasets/df_all.csv"
    s_df_all=requests.get(url_df_all).content
    df_all=pd.read_csv(io.StringIO(s_df_all.decode('utf-8')))
             
    fig, ax = plt.subplots()
    sns.countplot(df_all['event'], ax=ax)
    st.pyplot(fig)
    
    st.write(""" 
             
           Afin de donner un peu de largeur au dataset, nous avons ajouté des variables calculées 
           (ex :durées entre les évènements de vue, mise au panier, achat, prix, disponibilité des items).
             
             """)
    
    st.dataframe (df_all.head())
    
    
   

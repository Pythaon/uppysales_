# -*- coding: utf-8 -*-


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

    image = Image.open("https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.png")
    
    st.image(image)
    
    
    st.write("""
             Le coude n'est pas très franc mais apparait autour du nombre de cluster = 4
             """)
             
    image2 = Image.open(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\silhouette.png')
    
    st.image(image2) 

    image3 = Image.open(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\graph_clusters.png')
    st.image(image3) 
    
    
    st.write("""
             Le modèle a créé 4 clusters qui semblent alignés sur l'axe de la variable prix'
             
             
            Calcul de la moyenne des variables par cluster:
             """)
             
    image4 = Image.open(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\graph_variables.png')        
             
    
    st.image(image4)   

    st.write("""
             ● Le modèle a principalement créé les clusters sur la différence de prix entre les
                articles, ce qui est cohérent avec la représentation graphique qui regroupe les points
                le long de l’axe de prix.

            ● En revanche, nous ne connaissons pas la nature des produits sur le site, nous ne
                pouvons pas savoir si ce classement est pertinent ou non.

            ● La segmentation obtenue par apprentissage supervisé utilise la variable prix plus que
                les variables de performances de l’article (nombre de vues, mises au panier,
                transaction) tels qu’utilisés dans le scoring
             
             """)      
    
    
if choix_modele==models[1]:
     #@st.cache 
     st.write("écrire l'autre modèle")

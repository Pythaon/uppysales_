# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:20:35 2021

@author: Utilisateur
"""

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

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import neighbors
from sklearn import datasets
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

##------- ALL PAGE 
st.set_page_config(page_title="UpPySales App",page_icon="🎯",layout="centered",initial_sidebar_state="expanded")

page = st.sidebar.radio(label="Menu", options = ['1️⃣ Présentation',  '2️⃣ Segmentation visiteurs', '3️⃣ Clustering'])

### LOGO
@st.cache
def img():
    urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.PNG"
    image = Image.open(requests.get(urllogo, stream=True).raw)
    return image
    
image = img()
    
st.image(image, width=None)

st.title("""**Analyse de l'activité de e-commerce**""")


if page =='3️⃣ Clustering':
    
    st.header("**3️⃣ Clustering**")
    
    st.markdown("""
            Nous allons tester les modèles suivants:
                
                
                """)
            
    models = ['Kmeans', 'Clustering Mixte kmeans & ACH']

    choix_modele = st.radio("", options=models)



    if choix_modele ==models[0]:
        @st.cache
        def img():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.PNG"
            image = Image.open(requests.get(urllogo, stream=True).raw)
            return image

        image = img()

        st.image(image, width=None)
        
        st.write("""
             Le coude n'est pas très franc mais apparait autour du nombre de cluster = 4
             """)
         
        @st.cache
        def img2():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/silhouette.PNG"
            image2 = Image.open(requests.get(urllogo, stream=True).raw)
            return image2

        image = img2()

        st.image(image, width=None)    
    
        @st.cache
        def img3():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_clusers.PNG"
            image3 = Image.open(requests.get(urllogo, stream=True).raw)
            return image3
    
        image = img3()
    
        st.image(image, width=None)
        
        
        st.write("""
                 Le modèle a créé 4 clusters qui semblent alignés sur l'axe de la variable prix'
                 
                 
                Calcul de la moyenne des variables par cluster:
                 """)
        @st.cache
        def img4():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_clusers.PNG"
            image4 = Image.open(requests.get(urllogo, stream=True).raw)
            return image4
    
        image = img4()
    
        st.image(image, width=None)        
            
    
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
    

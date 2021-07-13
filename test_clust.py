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
    urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_coude.png""
    image = Image.open(requests.get(urllogo, stream=True).raw)
    return image
    
image = img()
    
st.image(image, width=None)

st.title("""**Analyse de l'activité de e-commerce**""")

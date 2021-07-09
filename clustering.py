# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:34:28 2021

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
from joblib import dump, load

# Enregistrement du modèle (à faire après l'entraînement)
#dump(model, 'nom_du_fichier.joblib') 
# Chargement du modèle (à faire sur l'app Streamlit)
#model = load('nom_du_fichier.joblib') 

page = st.sidebar.radio(label="Menu", options = ['Présentation',  'Segmentation visiteurs', 'Segmentation produits',
                                  'Clustering', 'test'])


df_all=pd.read_csv(r'C:\Users\Utilisateur\Documents\Data scientest\Projet\Dataset\df_all.csv')
df_all=df_all.iloc[:3000]


if page =='Clustering':
    
    st.title ("Clustering")
    
    
    
    
    items = df_all.groupby(df_all['itemid'], as_index = False).agg({'event':'count', 'ev_view':'sum','ev_addtocart':'sum', 'ev_transaction':'sum', 'price':'mean', 'categoryid':'mean', 'parentid':'mean'})
    
    
    
    # on retire les lignes sans prix
    items = items.dropna(axis = 0, how='all', subset=['price'])
    # suppression des variables catégorielles
    items = items.drop(['categoryid', 'parentid', 'itemid', 'event'], axis = 1)
    
    # Normalisation
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    items_sc = scaler.fit(items)
    items_sc = scaler.transform(items)
    
   
    
    #fonction qui lance les modèles
    

    st.markdown("""
                Nous allons tester les modèles suivants:""")
                
    models = ['Kmeans', 'Clustering Mixte kmeans & ACH']
              
    choix_modele = st.radio("", options=models)
    
    if choix_modele ==models[0]:
                 
        
        
        
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
        # Liste des nombre de clusters
        
        range_n_clusters = np.arange(2,10)
        
        # Initialisation de la liste de distortions
        distortions = []
        
        # Calcul des distortions pour les différents modèles
        for n_clusters in range_n_clusters:
            # Initialisation d'un cluster ayant un pour nombre de clusters n_clusters
            cluster = KMeans(n_clusters = n_clusters)
            # Apprentissage des données suivant le cluster construit ci-dessus
            cluster.fit(items)
            # Ajout de la nouvelle distortion à la liste des données
            distortions.append(sum(np.min(cdist(items_sc, cluster.cluster_centers_, 'euclidean'), axis=1)) / np.size(items, axis = 0))
        
        
            # Courbe du coude
        fig, ax = plt.subplots()
        #plt.figure(figsize=(5, 6))
        plt.plot(range_n_clusters, distortions)
        plt.xlabel('Nombre de Clusters K')
        plt.ylabel('Distortion (WSS/TSS)')
        plt.title('Méthode du coude affichant le nombre de clusters optimal')
        st.pyplot(fig)
        
    
        
        # Algorithme de K-means
        kmeans = KMeans(n_clusters = 4)
        kmeans.fit(items_sc)
    
        # Centroids and labels
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        items_sc = pd.DataFrame(items_sc)
        
        
        
        # Calcul du coefficient silhouette
        from sklearn.metrics import silhouette_score
        st.write("""Le coefficient de silhouette est de:""")
        st.write(silhouette_score(items_sc, labels, metric='sqeuclidean'))
        
       
        
       
        st.subheader("Représentation graphique des clusters")
        
        # Liste des coleurs
        fig, ax = plt.subplots()
        colors = ["g.","r.","c.","y.","b."]
        #plt.figure(figsize = (8,8))
        # Grphique du nuage de points attribués au cluster correspondant
        for i in range(len(items_sc)):
            plt.plot(items_sc.iloc[i,2], items_sc.iloc[i,3], colors[labels[i]], markersize = 10)
        plt.xlabel('transaction')
        plt.ylabel('price')
        st.pyplot(fig)
        # Graphique des centroïdes
        #plt.scatter(centroids[:, 0],centroids[:, 1], marker = "o", color = "blue",s=30, linewidths = 1, zorder = 10)
        
        
        #plt.xlabel('transaction')
        #plt.ylabel('price')
        #plt.title('clusters')
        #plt.show()

        
        # standardisation
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(items))
        df_scaled.columns = items.columns
        df_scaled['km_labels'] = labels
        
        # Calcul des moyennes de chaque variable pour chaque cluster
        df_mean = df_scaled.loc[df_scaled.km_labels!=-1, :].groupby('km_labels').mean().reset_index()
   
        # Représentation graphique

        results = pd.DataFrame(columns=['Variable', 'Std'])
        for column in df_mean.columns[1:]:
            results.loc[len(results), :] = [column, np.std(df_mean[column])]
        selected_columns = list(results.sort_values('Std', ascending=False).head(7).Variable.values) + ['km_labels']
        
        
        
        st.subheader("""Représentation graphique de l'importance des variables dans le clustering Kmeans""")
        # Graphique
        tidy = df_scaled[selected_columns].melt(id_vars='km_labels')
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.barplot(x='km_labels', y='value', hue='variable', data=tidy, palette='Set3')
        plt.legend(loc='upper right')
        plt.savefig("km_labels.jpg", dpi=300)
        st.pyplot(fig)
        
        
        st.write("""
                  Le modèle a principalement créé les clusters sur la différence de prix entre les
            articles, ce qui est cohérent avec la représentation graphique qui regroupe les points
            le long de l’axe de prix.
            
            
            En revanche, nous ne connaissons pas la nature des produits sur le site, nous ne
            pouvons pas savoir si ce classement est pertinent ou non.
            
            
            La segmentation obtenue par apprentissage supervisé utilise la variable prix plus que
            les variables de performances de l’article (nombre de vues, mises au panier,
            transaction) tels qu’utilisés dans le scoring
            d)
                 """)
        
    if choix_modele==models[1]:
        st.write("écrire l'autre modèle")
        
        
           
                
                
        
        
if page == 'test':
    
    fig, ax = plt.subplots()
    sns.countplot(df_all['ev_view'], ax=ax)
    st.pyplot(fig)
    
    

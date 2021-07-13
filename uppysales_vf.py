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

##------- PAGE PRÉSENTATION
if page == '1️⃣ Présentation':
        
    
    st.header("""**1️⃣ Présentation**""")  
             
    st.write(""" 
             
            Le jeu de données est un journal d'activité, il contient donc des timestamps, 
            des actions de visiteurs sur le site pendant 139 jours.
            Nous connaissons l'activité des visiteurs: l’article (itemid) visité, 
            le/les mis au panier, ou acheté(s), le prix de l'item et sa disponibilité.
            La nature des articles est anonymisée (nous ne connaissons pas le contenu du site).
             
             """)
    
    ### -- 
    st.subheader("""La table principale est la table event:""") 

    st.dataframe(events.head())
    
    st.write(""" 
             
            La table est très étroite avec seulement 5 variables, de plus le jeu de données est
            déséquilibré sur la variable **transactionid** avec moins de **1%** des enregistrements 
            qui concernent des transactions.
             
             """)
   
  
    fig, ax = plt.subplots(figsize=(4,3))
    sns.countplot(df_all['event'], ax=ax)
    #plt.figure(figsize=(1,3))
    st.pyplot(fig)
    
    ### -- 
    
    st.write(""" 
             
           Afin de donner un peu de largeur au dataset, nous avons ajouté des variables calculées 
           (ex :durées entre les évènements de vue, mise au panier, achat, prix, disponibilité des items).
             
             """)
   
  
    st.subheader("""La table ci-dessous est un sample random de 30% du dataset principal retravaillé""")   
    
    st.dataframe(df_all.head())

    
##------- PAGE SEGMENTATION VISITEURS 

if page == '2️⃣ Segmentation visiteurs':
        
    def main():
    
        st.header('**2️⃣ Segmentation visiteurs**')
        st.write("""
         La première analyse du jeu de données nous suggère que le site attire beaucoup de visiteurs
            mais réalise très peu de transactions. Nous pensons que pour éviter le biais du survivant, il
            faut éviter de se focaliser uniquement sur les comportements des clients qui ont réalisé une
            transaction, mais aussi observer les interactions qui n'ont pas donné lieu à une transaction
            pour pouvoir suggérer des changements au propriétaire de ce site e-commerce pour
            améliorer son taux de conversion.
            
            Ainsi notre analyse RFM prend en compte les éléments suivants:
                
                
            ● R - Récence : date de dernière action sur le site
        
            
            ● F - Fréquence : nombre d'action sur le site
            
            
            ● M - Monetary : Montant des transactions s’il y en a eu
        
        """)

        
        @st.cache
        def seg_vis(df_all):
        
            rev_par = df_all.groupby(['transactionid', 'visitorid']).agg({'price':'sum'}).reset_index()
            rev_par = rev_par.groupby('visitorid').agg({'price':'sum'}).reset_index()
            rev_par = rev_par.rename(columns={'price':'rev par visitor'})
            
            rev_par.head(2)
            
            #Fusion avec le dataset principal
            df_sco=df_all.merge(rev_par, how='left', on='visitorid')
            
            
            
              
            # Grouper
            visitors = df_sco.groupby(df_sco['visitorid'], as_index = False).agg({'timestamp':np.max,'event':'count',
                                                                              'ev_view':'sum','ev_addtocart':'sum',
                                                                              'ev_transaction':'sum', 'itemid':'count',
                                                                              'rev par visitor':np.max,'categoryid':'count',
                                                                              'parentid':'count',})
            
            # Formater
            visitors.visitorid=visitors.visitorid.astype('object')
            
            visitors["date_auj"] = datetime.date.today()
            visitors["date_max"] = df_sco['timestamp'].max()
            
            # Assignation des colonnes F et M du score RFM
            visitors =visitors.rename(columns={'event':'F'})
            visitors =visitors.rename(columns={'rev par visitor':'M'})
            
            
            visitors['date_max']=pd.to_datetime(visitors['date_max'])
            visitors['timestamp']=pd.to_datetime(visitors['timestamp'])
            
            # Création de la colonne R - écart de la date de dernière action avec la dernière date du dataset
            visitors["R"] = visitors.date_max - visitors.timestamp
            
            # Suppression des colonnes inutiles
            visitors= visitors.drop(['timestamp', 'ev_view', 'ev_addtocart', 'ev_transaction', 'itemid',
               'categoryid', 'parentid', 'date_auj', 'date_max'], axis=1)
            
            # Changement de type de la colonne R qui doit être un nombre de jour qui permet les calculs
            visitors['R']=visitors['R'].dt.days
            visitors['R']=visitors['R'].astype('int')
            
            # Statistiques des variables de la table visitors
            #visitors.describe().round(2).T
            
            # Analyse sur M hors 0 (pour pouvoir segmenter les acheteurs en fonction de leur valeur, le M = 0 qui
            # correspond à un chiffre d'affaires de 0€ des visiteurs sans achat étant prépondérant celui-doit être
            # exclu pour voir la répoartition de M pour les acheteurs et en faire le score))
            
            M_hors_0 = visitors[(visitors['M']!=0)]
            M_hors_0['M'].describe().round(2)
            
            # Création des scores grace aux quartiles
            # F : choix de Q2, Q3, max
            # Les valeurs retenues pour M sont 18 720 (Q1), 51 480  (Q3), 3 278 784 (max)
            r_bins = [-1, 35, 66, 137]
            f_bins = [0, 1, 2, 2465]
            m_bins = [0, 18720, 51480, 3278784]
            visitors['R_score'] = pd.cut(visitors['R'], r_bins, labels = ["3", "2", "1"])
            visitors['F_score'] = pd.cut(visitors['F'], f_bins, labels = ["1", "2", "3"])
            visitors['M_score'] = pd.cut(visitors['M'], m_bins, labels = ["1", "2", "3"])
               
            
            # Remplacement des NAN par valeur "0"
            # remplacement: ajouter une nouvelle catégorie puis fill na avec cette nouvelle catégorie (https://stackoverflow.com/questions/32718639/pandas-filling-nans-in-categorical-data/44633307)
            visitors['M_score'] = visitors['M_score'].cat.add_categories(0).fillna(0)
            
            # Concaténation du score RFM
            visitors["RFM_SCORE"] = visitors['R_score'].astype(str) + visitors['F_score'].astype(str) + visitors['M_score'].astype(str)
            
               
            # création du dictionnaire de référence score RFM => Segment
            dico = {'110':'0_Non intéressé','111':'1_Nouveaux ','112':'1_Nouveaux ','113':'1_Champions','120':'0_Curieux','121':'1_Nouveaux ','122':'1_Nouveaux ','123':'1_Champions','130':'0_Intéressé','131':'1_Nouveaux ','132':'1_Nouveaux ','133':'1_Champions','210':'0_Non intéressé','211':'1_A fidéliser','212':'1_A fidéliser','213':'1_Champions','220':'0_Curieux','221':'1_A fidéliser','222':'1_A fidéliser','223':'1_Champions','230':'0_Intéressé','231':'1_A fidéliser','232':'1_A fidéliser','233':'1_Champions','310':'0_Non intéressé','311':'1_A retenir','312':'1_A retenir','313':'1_Champions en risque','320':'0_Curieux','321':'1_A retenir','322':'1_A retenir','323':'1_Champions en risque','330':'0_Intéressé','331':'1_A retenir','332':'1_A retenir','333':'1_Champions en risque'}
            
            visitors['RFM_SCORE'] = visitors['RFM_SCORE'].astype('str')
            
            # Ajout de la colonne segment
            visitors['segment'] = visitors['RFM_SCORE'].map(dico)
            
            
            # Compte du nombre de visitorid par segment
            graph = visitors.groupby("segment").agg({'visitorid':'count'}).reset_index(0)
            somme=graph['visitorid'].sum()
            graph['percent'] = ((graph['visitorid']/somme)*100).round(2)
        
            return(graph)
        
        
        graph = seg_vis(df_all)       
        
        st.table(graph)
        
        status = st.radio("Selectionner la vue: ", ('Ensemble des visiteurs', 'Focus acheteurs'))
        
        if (status == 'Ensemble des visiteurs'):
       
            
            def status_1(graph):
            
                # Représentation graphique des segments
                fig, ax = plt.subplots()
                sns.set(rc={'figure.figsize':(20.7,15)})
                g = sns.catplot(x='segment',y='percent',kind='bar',data=graph, height=6, aspect= 3)
                g.ax.set_ylim(0,80)
                for p in g.ax.patches:
                    txt = str(p.get_height().round(0)) + '%'
                    txt_x = p.get_x()
                    txt_y = p.get_height()
                    g.ax.text(txt_x,txt_y,txt);
                    
                st.pyplot(g)
                
            status_1(graph)
            
            st.write("""
                         L'allure de la représentation graphique n'échappe pas à la nature désésquilibrée du dataset.
            
                Elle permet d'observer néamoins qu'on peut trouver 3 groupes de visiteurs sans achat :
                
                **> Non intéressé:**
                
                cette catégorie la plus importante concerne les visiteurs qui sont entrés sur le site et sortis immédiatement (pas de transaction, nombre d'action =1)
                
                - soit les clients sont entrés par erreur, ce qui peut suggérer que le site est mal référencé remonte en résultat dans une recherche autre que celle de l'objet du site.
                
                - Soit les clients ont voulu voir un article qui n'est pas disponible et sont ressortis immédiatement.
                
                Analyse complémentaire: parmis les visites des non intéressés quel % concernait des produits non disponibles?
                
                **> Intéressés et curieux:**
                
                sont des visiteurs qui ont parcouru le site, les interessés ayant fait deux actions ou plus. Ceci montre que 14% des visiteurs du site étaient intéressés par ce qui s'y vend mais non pas été jusqu'à la transaction. Ces clients sont un vivier potentiel de conversion.
                
                Analyse complémentaire: quels sont les attribus des produits consultés qui ont pu limiter la conversion?
                         """)
        
             
        else:
        
            def status_2(graph):
            
                graph_ach = graph.copy()
                
                # Filtre sur les catégories commençant par 1
                # Extraction du premier caractère de la colonne segment
                graph_ach['rep']=graph_ach['segment'].str[0]
                # Filtrage
                graph_ach= graph_ach[(graph_ach['rep']=='1')]
                graph_ach= graph_ach.drop(['rep', 'percent'], axis=1)
            
                
                # Compte du nombre de visitorid par segment
                somme_ach=graph_ach['visitorid'].sum()
                graph_ach['percent'] = ((graph_ach['visitorid']/somme_ach)*100).round(2)
            
                
                # Représentation graphique des segments
                sns.set(rc={'figure.figsize':(20.7,15)})
                
                g = sns.catplot(x='segment',y='percent',kind='bar',data=graph_ach, height=6, aspect= 3)
                g.ax.set_ylim(0,50)
                
                for p in g.ax.patches:
                    txt = str(p.get_height().round(0)) + '%'
                    txt_x = p.get_x()
                    txt_y = p.get_height()
                    g.ax.text(txt_x,txt_y,txt);
                    
                st.pyplot(g)
                
            status_2(graph)
            
            
        
            st.write("""
                    **40%** des acheteurs sont des acheteurs récents, les identifier permet de pouvoir leur adresser une campagne de bienvenue et de remerciement. Nous pouvons également tenter de les fidéliser.
                    
                    **17%** des acheteurs sont des "champions", en terme de chiffre d'affaires, mais attention 7% d'entre eux sont en risque car n'ont pas eu d'activité récente.
                    La segmentation "champion" appelle a un traitement plus dédié de ce segment de clientèle.
                    
                    **20%** des acheteurs sont "à retenir", leur chiffre d'affaires et de moyen à faible, mais ils no'nt pas eu d'activité récente, ils sont à relancer.
                    
                    **17%**des clients n'ont pas de caractéristiques particulères de recence ou de chiffre d'affaires, ils peuevnt être ciblés par une campagne de communication plus généraliste pour les faire revenir sur le site et les fidéliser.
                    
                    Il conviendrait de rejouer cette segmentation sur un jeu de données avec un journal d'activité du site sur une plus longue période, afin d'observer les comportements dans le temps de manière plus précise.
                    
                    Ici, sur une période de 4 mois il y a **40% des acheteurs** qui sont consédérés comme nouveaux, il est possible qu'une partie d'entre eux avaient fait des achats dans les semaines avant le début du relevé.
                                     """)
           
    main()    
    

   
             
##------- PAGE CLUSTERING 

if page =='3️⃣ Clustering':
    
    st.header("**3️⃣ Clustering**")
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
        
    def main2():

        if choix_modele ==models[0]:
            
            @st.cache(suppress_st_warning=True)
            def cluster(graph1):
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
                fig_coude, ax = plt.subplots()
                #plt.figure(figsize=(5, 6))
                plt.plot(range_n_clusters, distortions)
                plt.xlabel('Nombre de Clusters K')
                plt.ylabel('Distortion (WSS/TSS)')
                plt.title('Méthode du coude affichant le nombre de clusters optimal')
                
                st.pyplot(fig_coude)
                
            cluster()
    main2()
    
    def main3():
    
        if choix_modele==models[1]:
            
            @st.cache
            def test(a,b):
                c = a + b
                return c
            a=5
            b=7
            res=test(a,b)
            
            st.write("""test = """, res)
    
    main3()
                

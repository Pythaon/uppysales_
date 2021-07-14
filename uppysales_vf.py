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
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_clusters.PNG"
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
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_variables.PNG"
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
        
        st.write(""" ici est tésté un modèle mixte, avec une classification ascendante hierarchique pour connaitre le nombre de clusters, suivi d'un k means""")
        
        
        
        
        @st.cache
        def img():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/graph_dendo.PNG"
            image = Image.open(requests.get(urllogo, stream=True).raw)
            return image

        image = img()

        st.image(image, width=None)
        
        st.write("""
             Le dendogramme laisse suggérer qu'il y a 3 clusters
             """)
         
        @st.cache
        def img2():
            urllogo = "https://raw.githubusercontent.com/Pythaon/uppysales_/main/Graph_cluster3.PNG"
            image2 = Image.open(requests.get(urllogo, stream=True).raw)
            return image2

        image = img2()

        st.image(image, width=None)    
        
        st.error("""Nous voyons dans le graphique principalement 2 clusters au lieu de 3, le troisième cluster ne contient qu'un point, ce modèle n'est donc pas fonctionnel""")

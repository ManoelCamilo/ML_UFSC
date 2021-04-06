# Clusterizando com KMeans
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from numpy import std
import folium
import webbrowser
import os
from branca.element import Figure
from folium import plugins
from folium.plugins import HeatMapWithTime
import math


def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss) - 1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 2


m0 = folium.Map(location=[-13, -51], zoom_start=5, control_scale=True)

# Importyando o dataSet
dataset = pd.read_csv('KMEANS-COVID.csv',delimiter=';')
Z= dataset.iloc[:, [0, 1]].values
# Identifica os Outliers com base na quantidade de passageiros
data_mean = mean(Z[:,0])
data_std = std(Z[:,0])
cut_off = data_std * 3
lower_x, upper_x = data_mean - cut_off, data_mean + cut_off
print('imprimindo limites para Outliers')
print(lower_x)
print("=======")
print(upper_x)

datasetSemOutliers=dataset[dataset.iloc[:,0] < upper_x]
datasetComOutliers=dataset[dataset.iloc[:,0] >= upper_x]

X = datasetSemOutliers.iloc[:, [0, 1]].values

# Usando o Método Elbow para encontrar o Número ótimo de Clusters.
# Sem Outliers, o resultado = 3 clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


numero_otimo=optimal_number_of_clusters(wcss)
print('Numero Ótimo: '+str(numero_otimo))
plt.plot(range(1, 10), wcss)
plt.title('Método Elbow - Quantidade de Clusters Ideal')
plt.xlabel('Número de Clusters')
plt.ylabel('Within-clusters sum-of-squares (WCSS)')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

intPosicaoRotulo=-1
for i in range(0,len(y_kmeans)):
    intPosicaoRotulo = -1*intPosicaoRotulo

    #Para cada cor de cluster, os que receberam voos internacionais recebem icones diferentes
    if str(datasetSemOutliers.iloc[i, 3])=='S':
            #Marcador na plotagem do Kmeans
            strMarker='+'
            #Ícone para usar no mapa se recebeu voos internacionais
            strIcone="glyphicon glyphicon-globe"
    else:
        strMarker = '.'
        strIcone="glyphicon glyphicon-home"

    if(y_kmeans[i]==0):
        strColor='green'

    elif(y_kmeans[i]==1):
        strColor='red'

    elif (y_kmeans[i] == 2):
        strColor = 'orange'

    elif (y_kmeans[i] == 3):
        strColor = 'blue'

    elif (y_kmeans[i] == 4):
        strColor = 'brown'


    folium.Marker(location=[datasetSemOutliers.iloc[i, 6], datasetSemOutliers.iloc[i, 7]],
                  popup=(datasetSemOutliers.iloc[i, 2] + " - Contaminação no <u>" + str(datasetSemOutliers.iloc[i, 1]) + '</u>º dia'),
                  icon=folium.Icon(color=strColor, icon=strIcone), ).add_to(m0)

    plt.plot(datasetSemOutliers.iloc[i,0], datasetSemOutliers.iloc[i,1],markersize=8, color=strColor,marker=strMarker, fillstyle='full',markeredgewidth=1)
    plt.annotate(datasetSemOutliers.iloc[i,2], (datasetSemOutliers.iloc[i,0] + (intPosicaoRotulo*.06), datasetSemOutliers.iloc[i,1] + (intPosicaoRotulo*.1)), fontsize=12)


# Adicionando ao Mapa um eventual Estado Outlier
for i in range(0,len(datasetComOutliers)):
    folium.Marker(location=[datasetComOutliers.iloc[i, 6], datasetComOutliers.iloc[i, 7]],
                  popup=(datasetComOutliers.iloc[i, 2] + " - Contaminação no <u>" + str(
                      datasetComOutliers.iloc[i, 1]) + '</u>º dia'),
                  icon=folium.Icon(color="gray", icon="glyphicon glyphicon-flag"), ).add_to(m0)



plt.title('Sars-Cov-2 - Dia dO 1o Caso emn cada UF')
plt.xlabel('Passageiros Recebidos')
plt.ylabel('Dia de Contaminação')
plt.legend()
plt.show()



# Pega o Path da Aplcação para salvar o mapa na pasta correta
application_path = application_path = os.path.dirname(os.path.realpath(__file__))

Lista_Latitude_Longitude_Temporal = []
for i in range(2,29):
    temp=[]
    for index, instance in dataset[dataset['DIA_ORD'] < i].iterrows():
        temp.append([instance['LAT'],instance['LON']])
    Lista_Latitude_Longitude_Temporal.append(temp)

fig0=Figure(width=850,height=550)
fig0.add_child(m0)

#Plugin de Tela Cheia
plugins.Fullscreen(position="topright",title="Tela Cheia",title_cancel="Sair da Tela Cheia",force_separate_button=True,).add_to(m0)
minimap = plugins.MiniMap()
m0.add_child(minimap)

HeatMapWithTime(Lista_Latitude_Longitude_Temporal,radius=30,auto_play=True,position='bottomright').add_to(m0)
m0.save(application_path+"\mapaCovid-ML-UFSC.html")
webbrowser.open_new_tab(application_path+"\mapaCovid-ML-UFSC.html")

# 1. SETUP
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from minisom import MiniSom

# 2. CARGA DE DATOS
df = pd.read_csv('dataset_inquilinos.csv', index_col = 'id_inquilino')

df.columns = [
'horario', 'bioritmo', 'nivel_educativo', 'leer', 'animacion', 
'cine', 'mascotas', 'cocinar', 'deporte', 'dieta', 'fumador',
'visitas', 'orden', 'musica_tipo', 'musica_alta', 'plan_perfecto', 'instrumento'
]

# Eliminar columnas inecesarias
df = df.drop(columns=['leer', 'animacion', 'cine', 'dieta'])

#ordenar columnas por orden

def ordenar(Valor_Orden):
    df = pd.read_csv('dataset_inquilinos.csv', index_col = 'id_inquilino')
    df.columns = [
        'horario', 'bioritmo', 'nivel_educativo', 'leer', 'animacion', 
        'cine', 'mascotas', 'cocinar', 'deporte', 'dieta', 'fumador',
        'visitas', 'orden', 'musica_tipo', 'musica_alta', 'plan_perfecto', 'instrumento'
    ]
    if(Valor_Orden != None):
        df = df.sort_values(by=[Valor_Orden])

# 3. ONE HOT ENCODING
# Realizar el one-hot encoding
#transforma la data a vlores de 0 y 1
encoder = OneHotEncoder(sparse_output=False)
df_encoded = encoder.fit_transform(df)

# Obtener los nombres de las variables codificadas después de realizar el one-hot encoding
encoded_feature_names = encoder.get_feature_names_out()

# 3.1. PCA
pca = PCA(n_components=10)  # Ajustar el número de componentes principales según sea necesario
df_pca = pca.fit_transform(df_encoded)

# 4. MATRIZ DE SIMILIARIDAD
# Calcular la matriz de similaridad utilizando el punto producto
matriz_s = np.dot(df_encoded, df_encoded.T)
# 4.1. MATRIZ DE SIMILITUD PCA
matriz_s_pca = np.dot(df_pca, df_pca.T)


# Define el rango de destino
rango_min = -100
rango_max = 100

# Encontrar el mínimo y máximo valor en matriz_s
min_original = np.min(matriz_s)
max_original = np.max(matriz_s)

# Reescalar la matriz
matriz_s_reescalada = ((matriz_s - min_original) / (max_original - min_original)) * (rango_max - rango_min) + rango_min

# Pasar a Pandas
df_similaridad = pd.DataFrame(matriz_s_reescalada,
        index = df.index,
        columns = df.index)


# 5. BÚSQUEDA DE INQUILINOS COMPATIBLES
'''
Input:
* id_inquilinos: el o los inquilinos actuales DEBE SER UNA LISTA aunque sea solo un dato
* topn: el número de inquilinos compatibles a buscar

Output:
Lista con 2 elementos.
Elemento 0: las características de los inquilinos compatibles
Elemento 1: el dato de similaridad
'''

# 6. AUMENTO CON KMEANS

# Entrenar el modelo KMeans
def entrenar_kmeans(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # Establecer n_init explícitamente
    kmeans.fit(df_encoded)
    return kmeans

# Asignar inquilinos a clústeres
def asignar_inquilinos_a_clusters(kmeans_model):
    cluster_labels = kmeans_model.predict(df_encoded)
    return cluster_labels


# Encontrar inquilinos compatibles dentro del mismo clúster
def inquilinos_compatibles_con_kmeans(id_inquilinos, topn, cluster_labels):
    
    # Obtener el clúster de los inquilinos de referencia
    referencia_cluster = cluster_labels[id_inquilinos]
    
    # Filtrar los inquilinos que están en el mismo clúster que los de referencia
    inquilinos_en_mismo_cluster = []
    for valor in referencia_cluster:
        inquilinos_en_mismo_cluster.extend(np.where(cluster_labels == valor)[0])
    inquilinos_en_mismo_cluster = np.unique(inquilinos_en_mismo_cluster)

    # Calcular la similitud promedio entre los inquilinos en el mismo clúster
    similitud_promedio_cluster = df_similaridad.iloc[inquilinos_en_mismo_cluster, :].mean(axis=0)

    # Excluir los inquilinos de referencia
    similitud_promedio_cluster = similitud_promedio_cluster.drop(id_inquilinos)

    # Tomar los topn inquilinos más similares dentro del mismo clúster
    topn_inquilinos_cluster = similitud_promedio_cluster.nlargest(topn)

    # Obtener los registros de los inquilinos similares dentro del mismo clúster
    registros_similares_cluster = df.loc[topn_inquilinos_cluster.index]

    # Obtener los registros de los inquilinos de referencia
    registros_buscados = df.loc[id_inquilinos]

    # Concatenar los registros de referencia con los registros similares dentro del mismo clúster
    resultado_cluster = pd.concat([registros_buscados.T, registros_similares_cluster.T], axis=1)

    # Crear un objeto Series con la similitud de los inquilinos similares dentro del mismo clúster
    similitud_series_cluster = pd.Series(data=topn_inquilinos_cluster.values, index=topn_inquilinos_cluster.index, name='Similitud')

    # Devolver el resultado dentro del mismo clúster y el objeto Series de similitud
    return(resultado_cluster, similitud_series_cluster)


def inquilinos_compatibles_con_pca(id_inquilinos, topn):
    # Calcular la similitud utilizando la matriz de similaridad PCA
    similitud_promedio_pca = pd.DataFrame(matriz_s_pca, index=df.index, columns=df.index).loc[id_inquilinos].mean(axis=0)
    
    # Ordenar los inquilinos en función de su similitud promedio
    inquilinos_similares_pca = similitud_promedio_pca.sort_values(ascending=False)
    
    # Excluir los inquilinos de referencia (los que están en la lista)
    inquilinos_similares_pca = inquilinos_similares_pca.drop(id_inquilinos)
    
    # Tomar los topn inquilinos más similares
    topn_inquilinos_pca = inquilinos_similares_pca.head(topn)
    
    # Obtener los registros de los inquilinos similares
    registros_similares_pca = df.loc[topn_inquilinos_pca.index]
    
    # Obtener los registros de los inquilinos buscados
    registros_buscados = df.loc[id_inquilinos]
    
    # Concatenar los registros buscados con los registros similares en las columnas
    resultado = pd.concat([registros_buscados.T, registros_similares_pca.T], axis=1)
    
    # Crear un objeto Series con la similitud de los inquilinos similares encontrados
    similitud_series = pd.Series(data=topn_inquilinos_pca.values, index=topn_inquilinos_pca.index, name='Similitud')
    
    # Devolver el resultado y el objeto Series
    return resultado, similitud_series

def inquilinos_compatibles_con_knn(id_inquilinos, topn):

  # Verificar si todos los ID de inquilinos existen en la matriz de similaridad
  for id_inquilino in id_inquilinos:
    if id_inquilino not in df_similaridad.index:
      return 'Al menos uno de los inquilinos no encontrado'

  # Definir el valor de K
  k = 2

  # Crear el modelo KNN
  knn = KNeighborsClassifier(n_neighbors=k)

  # Convertir la matriz de NumPy a un Pandas DataFrame
  df_pca = pd.DataFrame(matriz_s_pca)

  # Entrenar el modelo con la matriz de similaridad y las etiquetas de los inquilinos
  knn.fit(df_pca, df.index)

  # Predecir la compatibilidad para los inquilinos de referencia
  predicciones = knn.predict_proba(df_pca.loc[id_inquilinos])

  # Obtener las probabilidades de compatibilidad para cada inquilino
  probabilidades = pd.DataFrame(predicciones, columns=df_similaridad.index)

  # Calcular la similitud promedio para cada inquilino
  similitud_promedio_knn = probabilidades.mean(axis=0)

  # Ordenar los inquilinos en función de su similitud promedio
  inquilinos_similares_knn = similitud_promedio_knn.sort_values(ascending=False)

  # Excluir los inquilinos de referencia (los que están en la lista)
  inquilinos_similares_knn = inquilinos_similares_knn.drop(id_inquilinos)

  # Tomar los topn inquilinos más similares
  topn_inquilinos_knn = inquilinos_similares_knn.head(topn)

  # Obtener los registros de los inquilinos similares
  registros_similares_knn = df.loc[topn_inquilinos_knn.index]

  # Obtener los registros de los inquilinos buscados
  registros_buscados = df.loc[id_inquilinos]

  # Concatenar los registros buscados con los registros similares en las columnas
  resultado_knn = pd.concat([registros_buscados.T, registros_similares_knn.T], axis=1)

  # Crear un objeto Series con la similitud de los inquilinos similares encontrados
  similitud_series_knn = pd.Series(data=topn_inquilinos_knn.values, index=topn_inquilinos_knn.index, name='Similitud')

  # Devolver el resultado y el objeto Series
  return resultado_knn, similitud_series_knn

import pandas as pd
from minisom import MiniSom
import numpy as np

def inquilinos_compatibles_con_som(id_inquilinos, topn, som_shape=(5, 5), num_epochs=100, learning_rate=0.1):

  """
  Encuentra inquilinos compatibles con los inquilinos de referencia usando un SOM.

  Args:
    id_inquilinos: Lista de IDs de los inquilinos de referencia.
    topn: Número de inquilinos más compatibles a encontrar.
    som_shape: Tupla con la forma de la grilla SOM (filas, columnas).
    num_epochs: Número de épocas de entrenamiento del SOM.
    learning_rate: Tasa de aprendizaje inicial del SOM.

  Returns:
    resultado_som: DataFrame con los registros de los inquilinos buscados y los 
      inquilinos similares encontrados dentro del mismo SOM.
    similitud_series_som: Serie con la similitud de los inquilinos similares dentro del mismo SOM.
  """

  # Verificar si todos los ID de inquilinos existen en la matriz de similaridad
  for id_inquilino in id_inquilinos:
    if id_inquilino not in df_similaridad.index:
      return 'Al menos uno de los inquilinos no encontrado'

  # Crear e inicializar el SOM
  som = MiniSom(som_shape[0], som_shape[1], df_similaridad.shape[1], sigma=1.0, learning_rate=learning_rate)
  som.random_weights_init(df_similaridad.values)

  # Entrenar el SOM
  som.train(df_similaridad.values, num_epochs, random_order=True)

  # Reshape id_inquilinos to match rows in df_similaridad
  id_inquilinos_reshaped = np.reshape(id_inquilinos, (-1, 1))  # Reshape to (n_tenants, 1)

  # Find tenants matching any reference tenant (if desired)
  # misma_posicion_som = df_similaridad.index[np.all(df_similaridad.values == df_similaridad.iloc[id_inquilinos_reshaped].values, axis=1).tolist()]

  # Find tenants matching the first reference tenant (common scenario)
  misma_posicion_som = df_similaridad.index[np.all(df_similaridad.values == df_similaridad.iloc[id_inquilinos_reshaped[0]].values, axis=1)]

  # Excluir los inquilinos de referencia
  inquilinos_en_mismo_som = misma_posicion_som.difference(set(id_inquilinos))

  # Calcular la similitud promedio entre los inquilinos en el mismo SOM
  similitud_promedio_som = df_similaridad.iloc[inquilinos_en_mismo_som, :].mean(axis=0)

  # Tomar los topn inquilinos más similares dentro del mismo SOM
  topn_inquilinos_som = similitud_promedio_som.nlargest(topn)

  # Obtener los registros de los inquilinos similares dentro del mismo SOM
  registros_similares_som = df.loc[topn_inquilinos_som.index]

  # Obtener los registros de los inquilinos de referencia
  registros_buscados = df.loc[id_inquilinos]

  # Concatenar los registros de referencia con los registros similares dentro del mismo SOM
  resultado_som = pd.concat([registros_buscados.T, registros_similares_som.T], axis=1)

  # Crear un objeto Series con la similitud de los inquilinos similares dentro del mismo SOM
  similitud_series_som = pd.Series(data=topn_inquilinos_som.values, index=topn_inquilinos_som.index, name='Similitud')

  # Devolver el resultado dentro del mismo SOM y el objeto Series de similitud
  return resultado_som, similitud_series_som




def inquilinos_compatibles(id_inquilinos, topn):
    # Verificar si todos los ID de inquilinos existen en la matriz de similaridad
    for id_inquilino in id_inquilinos:
        if id_inquilino not in df_similaridad.index:
            return 'Al menos uno de los inquilinos no encontrado'

    # Obtener las filas correspondientes a los inquilinos dados
    filas_inquilinos = df_similaridad.loc[id_inquilinos]

    # Calcular la similitud promedio entre los inquilinos
    similitud_promedio = filas_inquilinos.mean(axis=0)

    # Ordenar los inquilinos en función de su similitud promedio
    inquilinos_similares = similitud_promedio.sort_values(ascending=False)

    # Excluir los inquilinos de referencia (los que están en la lista)
    inquilinos_similares = inquilinos_similares.drop(id_inquilinos)

    # Tomar los topn inquilinos más similares
    topn_inquilinos = inquilinos_similares.head(topn)

    # Obtener los registros de los inquilinos similares
    registros_similares = df.loc[topn_inquilinos.index]

    # Obtener los registros de los inquilinos buscados
    registros_buscados = df.loc[id_inquilinos]

    # Concatenar los registros buscados con los registros similares en las columnas
    resultado = pd.concat([registros_buscados.T, registros_similares.T], axis=1)

    # Crear un objeto Series con la similitud de los inquilinos similares encontrados
    similitud_series = pd.Series(data=topn_inquilinos.values, index=topn_inquilinos.index, name='Similitud')

    # Devolver el resultado y el objeto Series
    return(resultado, similitud_series)


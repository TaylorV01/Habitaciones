import streamlit as st
import pandas as pd
from logica import inquilinos_compatibles
#KMEANS
from logica import inquilinos_compatibles_con_kmeans, asignar_inquilinos_a_clusters, entrenar_kmeans, ordenar, inquilinos_compatibles_con_som
from ayudantes import generar_grafico_compatibilidad, generar_tabla_compatibilidad, obtener_id_inquilinos
#PCA
from logica import inquilinos_compatibles_con_pca
#KNN
from logica import inquilinos_compatibles_con_knn

# Configurar la página para utilizar un layout más amplio.

st.set_page_config(layout="wide")

resultado = None
resultado1 = None
resultado_pca = None
resultado_knn = None
resultado_SOM = None

# Mostrar una gran imagen en la parte superior.
st.image('./Media/portada.png', use_column_width=True)

# Insertar un espacio vertical de 60px
st.markdown(f'<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)

# Configurar el sidebar con inputs y un botón.
with st.sidebar:
    st.header("¿Quién está viviendo ya en el piso?")
    inquilino1 = st.text_input("Inquilino 1")
    inquilino2 = st.text_input("Inquilino 2")
    inquilino3 = st.text_input("Inquilino 3")
    
    num_compañeros = st.text_input("¿Cuántos nuevos compañeros quieres buscar?")
    
    # Uso de KMEANS
    kmeans_model = entrenar_kmeans(10)  # ajusta el número de clústeres según sea necesario
    cluster_labels = asignar_inquilinos_a_clusters(kmeans_model)
    
    Valor_Orden = st.selectbox(
        "Selecciona el criterio de mayor importancia:",
        options=["horario", "bioritmo", "nivel_educativo", "leer", "animacion", 
        "cine", "mascotas", "cocinar", "deporte", "dieta", "fumador",
        "visitas", "orden", "musica_tipo", "musica_alta", "plan_perfecto", "instrumento"])
    
    
    if st.button('BUSCAR NUEVOS COMPAÑEROS'):
        # Verifica que el número de compañeros sea un valor válido
        try:
            topn = int(num_compañeros)
        except ValueError:
            st.error("Por favor, ingresa un número válido para el número de compañeros.")
            topn = None
        
        # Obtener los identificadores de inquilinos utilizando la función
        id_inquilinos = obtener_id_inquilinos(inquilino1, inquilino2, inquilino3, topn)

        if id_inquilinos and topn is not None:
            # Llama a la función inquilinos_compatibles con los parámetros correspondientes
            resultado = inquilinos_compatibles(id_inquilinos, topn)
            ordenar(Valor_Orden)
            #Esta linea es para mostrar el KMEANS
            resultado1 =inquilinos_compatibles_con_kmeans(id_inquilinos, topn, cluster_labels) 
            #Esta linea es para mostrar el PCA
            resultado_pca = inquilinos_compatibles_con_pca(id_inquilinos, topn)
            #Esta linea es para mostrar el KNN
            resultado_knn = inquilinos_compatibles_con_knn(id_inquilinos,topn)
            #Linea para mostrar SOM
            resultado_SOM = inquilinos_compatibles_con_som(id_inquilinos,topn)

# Verificar si 'resultado' contiene un mensaje de error (cadena de texto)
if isinstance(resultado, str):
    st.error(resultado)

# Si no, y si 'resultado' no es None, mostrar el gráfico de barras y la tabla
elif resultado is not None:
    st.header("RESULTADO NUMERO UNO APLICANDO EL METODO INICIAL")
    cols = st.columns((2))  # Divide el layout en 2 columnas
    
    with cols[0]:  # Esto hace que el gráfico y su título aparezcan en la primera columna
        st.write("Nivel de compatibilidad de cada nuevo compañero:")
        fig_grafico = generar_grafico_compatibilidad(resultado[1])
        st.pyplot(fig_grafico)
    
    with cols[1]:  # Esto hace que la tabla y su título aparezcan en la segunda columna
        st.write("Comparativa entre compañeros:")
        fig_tabla = generar_tabla_compatibilidad(resultado)
        st.plotly_chart(fig_tabla, use_container_width=True)


# Verificar si 'resultado' contiene un mensaje de error (cadena de texto)
if isinstance(resultado1, str):
    st.error(resultado1)

# Si no, y si 'resultado1' no es None, mostrar el gráfico de barras y la tabla
elif resultado1 is not None:
    st.header("RESULTADO APLICANDO EL ALGORITMO DE K_MEANS")
    cols = st.columns((2))  # Divide el layout en 2 columnas
    
    with cols[0]:  # Esto hace que el gráfico y su título aparezcan en la primera columna
        st.write("Nivel de compatibilidad de cada nuevo compañero:")
        fig_grafico = generar_grafico_compatibilidad(resultado1[1])  # Modificado para usar la similitud correcta
        #st.plotly_chart(fig_grafico, use_container_width=True, width=800, height=600, config={'displayModeBar': False})
        st.pyplot(fig_grafico)

    with cols[1]:  # Esto hace que la tabla y su título aparezcan en la segunda columna
        st.write("Comparativa entre compañeros:")
        fig_tabla = generar_tabla_compatibilidad(resultado1)  # Modificado para mostrar la información correcta
        st.plotly_chart(fig_tabla, use_container_width=True)


# Verificar si 'resultado' contiene un mensaje de error (cadena de texto)
if isinstance(resultado_pca, str):
    st.error(resultado_pca)

# Si no, y si 'resultado_pca' no es None, mostrar el gráfico de barras y la tabla
elif resultado_pca is not None:
    st.header("RESULTADO APLICANDO EL ALGORITMO DE PCA")
    cols = st.columns((2))  # Divide el layout en 2 columnas
    
    with cols[0]:  # Esto hace que el gráfico y su título aparezcan en la primera columna
        st.write("Nivel de compatibilidad de cada nuevo compañero:")
        fig_grafico = generar_grafico_compatibilidad(resultado_pca[1])  # Modificado para usar la similitud correcta
        #st.plotly_chart(fig_grafico, use_container_width=True, width=800, height=600, config={'displayModeBar': False})
        st.pyplot(fig_grafico)

    with cols[1]:  # Esto hace que la tabla y su título aparezcan en la segunda columna
        st.write("Comparativa entre compañeros:")
        fig_tabla = generar_tabla_compatibilidad(resultado_pca)  # Modificado para mostrar la información correcta
        st.plotly_chart(fig_tabla, use_container_width=True)

# Verificar si 'resultado' contiene un mensaje de error (cadena de texto)
if isinstance(resultado_SOM, str):
    st.error(resultado_SOM)

# Si no, y si 'resultado_SOM' no es None, mostrar el gráfico de barras y la tabla
elif resultado_SOM is not None:
    st.header("RESULTADO APLICANDO EL ALGORITMO DE SOM")
    cols = st.columns((2))  # Divide el layout en 2 columnas
    
    with cols[0]:  # Esto hace que el gráfico y su título aparezcan en la primera columna
        st.write("Nivel de compatibilidad de cada nuevo compañero:")
        fig_grafico = generar_grafico_compatibilidad(resultado_SOM[1])  # Modificado para usar la similitud correcta
        #st.plotly_chart(fig_grafico, use_container_width=True, width=800, height=600, config={'displayModeBar': False})
        st.pyplot(fig_grafico)

    with cols[1]:  # Esto hace que la tabla y su título aparezcan en la segunda columna
        st.write("Comparativa entre compañeros:")
        fig_tabla = generar_tabla_compatibilidad(resultado_SOM)  # Modificado para mostrar la información correcta
        st.plotly_chart(fig_tabla, use_container_width=True)
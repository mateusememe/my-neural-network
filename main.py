import streamlit as st
import pandas as pd
import numpy as np
import csv

def treinar_rede():
    print("treinando rede")


st.write("""## My neural network""")


##setup GUI

#numero de neuronios
linha1 = st.columns((1,1,1))
with linha1[0]:
    input_layer = st.number_input("Camada de Entrada:", min_value=1, step=1)
with linha1[1]:
    output_layer = st.number_input("Camada de Saída:", min_value=1, step=1)
with linha1[2]:
    hidden_layer = st.number_input("Camada de Oculta:", min_value=1, step=1)

#valor do erro
error_value = st.number_input("Valor do erro:", format="%.4f", min_value=0.0001, step=0.0001)


linha3 = st.columns((1,1))
with linha3[0]:
    #número de iterações
    iterations_num = st.number_input('Número de iterações:', min_value=100, step=100)
with linha3[1]:
    #função de transferência
    transfer_function = st.radio("Função de transferência",('Linear', 'Logística', 'Hiperbólica'))


uploaded_file = st.file_uploader("Arquivo de treinamento", 'csv', help='Faça upload de um arquivo csv para treinar a rede neural')

show_file = st.empty()
if uploaded_file is None:
    show_file.info("")
else:
    ##to know type of file
    #type = from_buffer(uploaded_file.getvalue())
    show_file.info(uploaded_file.name)
    dataframe = pd.read_csv(uploaded_file)
    input_size = len(dataframe.columns)-1
    columns = dataframe.columns[:input_size]
    print(columns)
    checkbox = []
    for col in columns:
        checkbox.append(st.checkbox(col, value=1))
    st.write(dataframe)

button =  st.button('Treinar rede', callable=treinar_rede)

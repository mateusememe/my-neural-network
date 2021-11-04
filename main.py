import streamlit as st
import pandas as pd
import numpy as np
import csv
import NeuralNetwork

global dataframe
dataframe = None


def treinar_rede(Data = None):
    Net = NeuralNetework()
    print("Data: ", Data)

st.write("""## My neural network""")


## Setup GUI

# Número de neuronios
linha1 = st.columns((1,1,1))
with linha1[0]:
    input_layer = st.number_input("Camada de Entrada:", min_value=1, step=1)
with linha1[1]:
    output_layer = st.number_input("Camada de Saída:", min_value=1, step=1)
with linha1[2]:
    hidden_layer = st.number_input("Camada de Oculta:", min_value=1, step=1)

# Valor do erro
error_value = st.number_input("Valor do erro:", format="%.4f", min_value=0.0001, step=0.0001)


linha3 = st.columns((1,1))
with linha3[0]:
    # Número de iterações
    iterations_num = st.number_input('Número de iterações:', min_value=100, step=100)
with linha3[1]:
    # Função de transferência
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
    input_size = len(dataframe.columns) - 1
    tupla_ones = tuple(np.ones(input_size))
                       
    checkbox_row = st.columns(tupla_ones)
    
    columns = dataframe.columns[:input_size]
    print(columns)
    checkbox = []
    i = 0
    for col in columns:
        with checkbox_row[i]:
            i = i + 1
            checkbox.append({"checked": st.checkbox(col, value=1, key=col, help="Select or no this attribute to use in network"), "attribute": col})
       
    
    for check in checkbox:
        if not(check["checked"]):
            dataframe = dataframe.drop(columns=[check['attribute']])
            print(dataframe.columns)
    st.dataframe(dataframe)
    
    button =  st.button('Treinar rede', on_click=treinar_rede, kwargs={'Data':dataframe})

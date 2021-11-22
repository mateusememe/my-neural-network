import streamlit as st
import pandas as pd
import numpy as np
import csv
from network import Network

global dataframe
dataframe = None


st.write("""# My Neural network""")
st.caption("An application of Multilayer Perceptron and backpropagation")

def train_test(data_train, data_test, learning, epochs=200, error=0.1, mode=1, n_hiddens=1):
    mlp = Network(epochs=int(epochs), error=float(error), mode=mode, n_hiddens=None)
    mlp.train(dataframe_train)

## Setup GUI

# Número de neuronios
linha1 = st.columns((1,1,1))
with linha1[0]:
    input_layer = st.text_input("Camada de Entrada:")
with linha1[1]:
    output_layer = st.text_input("Camada de Saída:")
with linha1[2]:
    hidden_layer = st.number_input("Camada de Oculta:", min_value=1, step=1)

# Valor do erro
error_value = st.number_input("Valor do erro:", format="%.4f", min_value=0.0001, step=0.0001)


linha3 = st.columns((1,1))
with linha3[0]:
    # Número de iterações
    iterations_num = st.number_input('Número de épocas:', min_value=100, step=100)
with linha3[1]:
    # Função de transferência
    transfer_function = st.radio("Função de transferência",('Linear', 'Logística', 'Hiperbólica'))

st.text("                                                              ")
st.text("                                                              ")
st.text("                                                              ")
linha2 = st.columns((1,1))
with linha2[0]:
    train_file = st.file_uploader("Arquivo de Treinamento", 'csv', help='Faça upload de um arquivo csv para treinar a rede neural')
with linha2[1]:
    test_file = st.file_uploader("Arquivo de Teste", 'csv', help='Faça upload de um arquivo csv para testar a rede neural')

show_file = st.empty()
if train_file is None and test_file is None:
    show_file.info("")
elif not(train_file is None) and not(test_file is None):
    ##to know type of file
    #type = from_buffer(train_file.getvalue())
    st.subheader("Base de Treinamento")
    show_file.info(train_file.name)
    st.subheader("Base de Teste")
    show_file.info(test_file.name)
    dataframe_train = pd.read_csv(train_file)
    dataframe_test = pd.read_csv(test_file)
    input_size = len(dataframe_train.columns) - 1
    tupla_ones = tuple(np.ones(input_size))
                       
    checkbox_row = st.columns(tupla_ones)
    
    columns = dataframe_test.columns[:input_size]
    print(columns)
    checkbox = []
    i = 0
    for col in columns:
        with checkbox_row[i]:
            i = i + 1
            checkbox.append({"checked": st.checkbox(col, value=1, key=col, help="Select or no this attribute to use in network"), "attribute": col})
       
    
    for check in checkbox:
        if not(check["checked"]):
            dataframe_train = dataframe_train.drop(columns=[check['attribute']])
            dataframe_test = dataframe_test.drop(columns=[check['attribute']])
    # columns length is equal for the datasets validation
    if len(dataframe_train.columns) == len(dataframe_test.columns):
        with st.expander('Dados de Treino:'):
            st.dataframe(dataframe_train)
        with st.expander('Dados de Teste:'):
            st.dataframe(dataframe_test)
    else:
        st.text("Entrada Inválida!")
    
    if st.button('Treinar rede'):
        
        print("\nDataset Train:\n", dataframe_train)
        print("\n\nDataset Test:\n", dataframe_test)
    
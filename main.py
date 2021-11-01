import streamlit as st
import pandas as pd
import numpy as np

st.write("""## My neural network""")

train = pd.read_csv('dataset/base_treinamento.csv', sep=',')

st.dataframe(train)

train.classe.apply(str)

print(train.dtypes)

st.line_chart(train)
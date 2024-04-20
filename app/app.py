import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import statsmodels.formula.api as smf

from funcoes import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.stats import ks_2samp


cwd = os.getcwd()
pipeline = cwd + '/pipeline_preprocessamento.pkl'
modelo_final = cwd + '/model_final.pkl'


def main():

    st.set_page_config(page_title = 'Regressão Logística - Pycaret', layout="wide", initial_sidebar_state='expanded')

    st.markdown('<div style="text-align: center; font-weight: 600; font-size: 2.25rem;">Regressão Logística</div>', unsafe_allow_html=True)
    arquivo = st.file_uploader('Enviar arquivo de dados:',  accept_multiple_files=False)

    preprocessamento = pickle.load(open(pipeline, 'rb'))
    modelo = pickle.load(open(modelo_final, 'rb'))
    
    if (arquivo != None):
        df = pd.read_feather(arquivo)
        df = df.sample(1000)
        df = preprocessamento.transform(df)

        X = df.copy()
        X.drop(columns=['mau'], inplace=True)
        y = df['mau']

        prev = modelo.predict(X)

        cp = X.copy()
        cp['mau'] = y
        cp['Previsão'] = prev

        st.write(cp)

if __name__ == '__main__':
	main()
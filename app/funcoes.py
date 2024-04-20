import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def imput_mean(data: pd.DataFrame):
    mean = data['tempo_emprego'].mean()
    data.loc[data.tempo_emprego.isna(), 'tempo_emprego'] = mean
    return data

def outliers(data: pd.DataFrame):
    data.drop(data.query('qtd_filhos >= 5').index, inplace=True)
    data.drop(data.query('tipo_renda == "Bolsista"').index, inplace=True)
    data['estado_civil'].replace({'Separado': 'Solteiro', 'União': 'Casado'}, inplace=True)
    data['educacao'].replace({'Superior incompleto': 'Superior','Superior completo': 'Superior'}, inplace=True)
    data['educacao'].replace({'Pós graduação': 'Superior'}, inplace=True)
    return data

def feature(data: pd.DataFrame):
    X = data.copy()
    y = X['mau']
    X.drop(columns=['mau', 'data_ref', 'index'], inplace=True, errors='ignore')
    
    forest = RandomForestClassifier(random_state=0, n_jobs=-1)
    forest.fit(X, y)
    
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=list(X.columns))
    
    remove = list(forest_importances.sort_values(ascending=False)[20:].index)
    X.drop(columns=remove, inplace=True, errors='ignore')
    X['mau'] = y
    return X

def fpca(data: pd.DataFrame):
    X_train = data
    y = X_train['mau']
    train_features = X_train

    # criando o PCA
    model = PCA(n_components=5).fit(train_features)
    X_pc = model.transform(train_features)

    # número de componentes
    n_pcs= model.components_.shape[0]

    # obtenha o índice do recurso mais importante em CADA componente, ou seja, o maior valor absoluto
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    initial_feature_names = X_train.columns.tolist()

    # recuperando os nomes
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # removendo as colunas não selecionadas
    remove = [item for item in list(data.columns) if item not in most_important_names]
    data.drop(columns=remove, inplace=True, errors='ignore')
    data['mau'] = y
    return data

def dummies(data: pd.DataFrame):
    data['renda'] = np.log(data['renda'])
    data['mau'] = data['mau'].astype('int')
    return pd.get_dummies(data, drop_first=True)
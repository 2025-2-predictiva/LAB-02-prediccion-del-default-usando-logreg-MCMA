# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



import pandas as pd
# lectura base de datos
train = pd.read_csv('../files/input/train_default_of_credit_card_clients.csv')
# renombreee la columa
train = train.rename(columns={'default payment next month':"default"})
# eliminee la varaible ID
train = train.drop(columns='ID')
# se verifican datos no diponibles 
#print(train.isnull().sum()) # raro hay datos extraños no se si se eliminan

columns_to_filter = ['MARRIAGE', 'EDUCATION'] 
#columns_to_filter1 = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] 
columns_to_filter1 = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

# se filtran para mayores a cero 
train = train[(train[columns_to_filter] > 0).all(axis=1)]

#train = train[(train[columns_to_filter] > 0).all(axis=1) & (train[columns_to_filter1] > 0).all(axis=1)]


# se agrupan niveles de educacion mayores a 4
train['EDUCATION'] = train['EDUCATION'].apply(lambda x: 'others' if x > 4 else x)


# lectura base de datos
test = pd.read_csv('../files/input/test_default_of_credit_card_clients.csv')
# renombrar la columna
test = test.rename(columns={'default payment next month':"default"})
# eliminoo la variable ID
test = test.drop(columns='ID')
# verificar datos no disponibles
#print(test.isnull().sum())  # revisar si hay datos extraños

# se filtran para mayores a cero
test = test[(test[columns_to_filter] > 0).all(axis=1)]

# agrupar niveles de educación mayores a 4
test['EDUCATION'] = test['EDUCATION'].apply(lambda x: 'others' if x > 4 else x)

X_train = train.drop(columns='default')
y_train = train['default']

X_test = test.drop(columns='default')
y_test = test['default']


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


categorical_features =  [
    'SEX',        # Género
    'EDUCATION',  # Educación
    'MARRIAGE',   # Estado civil
    'PAY_0',      # Estado del pago en septiembre
    'PAY_2',      # Estado del pago en agosto
    'PAY_3',      # Estado del pago en julio
    'PAY_4',      # Estado del pago en junio
    'PAY_5',      # Estado del pago en mayo
    'PAY_6'       # Estado del pago en abril
]

X_train[categorical_features] = X_train[categorical_features].astype(str)
X_test[categorical_features] = X_test[categorical_features].astype(str)
numerical_features = [col for col in X_train.columns if col not in categorical_features]
numerical_features = [col for col in X_test.columns if col not in categorical_features]

# Crear el preprocesador utilizando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

# pipe line con random forest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=666,bootstrap=True,class_weight='balanced',
                                          criterion='gini'))
])

# Entrenar y evaluar
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# parametros random forest a modificar
# parametros random forest a modificar
param_grid = {
    'classifier__n_estimators': [180],
    'classifier__max_depth': [None],
    'classifier__max_features':  ['sqrt'],
    'classifier__min_samples_split': [10],
    'classifier__min_samples_leaf': [2]
    
}

# Realizar GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1,verbose=True,refit=True)
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("Mejores parámetros:", grid_search.best_params_)


import joblib
import gzip
import os

# Directorio y archivo del modelo
output_dir = '../files/models/'
output_file = os.path.join(output_dir, 'model.pkl.gz')

# Cargar el modelo entrenado
with gzip.open(output_file, 'rb') as f:
    model = joblib.load(f)



import json
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
import os
import gzip
import joblib

# Cargar el modelo guardado
output_dir = '../files/models/'
output_file = os.path.join(output_dir, 'model.pkl.gz')

with gzip.open(output_file, 'rb') as f:
    best_model = joblib.load(f)

# Predicciones para entrenamiento y prueba
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Cálculo de las métricas para el conjunto de entrenamiento
metrics_train = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}

# Cálculo de las métricas para el conjunto de prueba
metrics_test = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

# Guardar las métricas en un archivo JSON
metrics = [metrics_train, metrics_test]
output_metrics_file = '../files/output/metrics.json'

# Crear el directorio de salida si no existe
os.makedirs(os.path.dirname(output_metrics_file), exist_ok=True)

# Escribir el archivo JSON
with open(output_metrics_file, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Métricas guardadas en: {output_metrics_file}")


import json
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import os
import gzip
import joblib

# Cargar el modelo guardado
output_dir = '../files/models/'
output_file = os.path.join(output_dir, 'model.pkl.gz')

with gzip.open(output_file, 'rb') as f:
    best_model = joblib.load(f)

# Predicciones para entrenamiento y prueba
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Cálculo de las métricas para el conjunto de entrenamiento
metrics_train = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}

# Cálculo de las métricas para el conjunto de prueba
metrics_test = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

# Cálculo de las matrices de confusión
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

cm_train_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
    'true_1': {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
}

cm_test_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
    'true_1': {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
}

# Guardar las métricas en un archivo JSON línea por línea
output_metrics_file = '../files/output/metrics.json'

# Crear el directorio de salida si no existe
os.makedirs(os.path.dirname(output_metrics_file), exist_ok=True)

# Escribir cada métrica y matriz en una línea independiente
with open(output_metrics_file, 'w', encoding='utf-8') as f:
    for entry in [metrics_train, metrics_test, cm_train_dict, cm_test_dict]:
        f.write(json.dumps(entry) + '\n')

print(f"Métricas y matrices de confusión guardadas en: {output_metrics_file}")


# combinacion con sobreajuste

#param_grid = {
#    'classifier__n_estimators': [100,102],
#    'classifier__max_depth': [None,5],
#    'classifier__max_features':  ['sqrt', 'log2'],
#}
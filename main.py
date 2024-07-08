# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Configurar el estilo de visualización
sns.set(style="whitegrid", palette="pastel")

# Especificar la ruta del archivo CSV en Kaggle
archivo_csv = 'C:/Users/juan.cortes/Desktop/Seminario/Proyecto/data/Churn_Modelling.csv'
# Ruta en Kaggle

# Leer el archivo CSV usando Pandas
datos = pd.read_csv(archivo_csv)

# Mostrar los primeros registros para verificar la carga correcta
print(datos.head())
print("El número total de registros/filas presentes en el conjunto de datos es",datos.shape[0])
print("El número total de registros/columnas presentes en el conjunto de datos es:",datos.shape[1])

print(datos.columns)





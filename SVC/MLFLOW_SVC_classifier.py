import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np

import mlflow
import mlflow.sklearn

def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    df.head()
    return df

def split_dataset(df: pd.DataFrame) -> np.ndarray:
    previsores = df.iloc[:, 0:20].values
    classe = df.iloc[:, 20].values
    X_train, X_test, y_train, y_test = train_test_split(
        previsores,
        classe,
        test_size=0.3,
        random_state=123)
    
    return X_train, X_test, y_train, y_test

def train_classifier(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:

    mlflow.set_experiment("svc_experimento")

    with mlflow.start_run():
        mlflow.autolog()

        mlflow.set_tag("model", "Support Vector Classifier (SVC)")
        mlflow.set_tag("author", "Felipe Amaral")
        mlflow.set_tag("objective", "Avaliar performance inicial do modelo SVC para o dataset Credit")

        model = SVC()
        model.fit(X_train, y_train)
        predicts = model.predict(X_test)

        # Métricas
        acuracia = accuracy_score(y_test, predicts)
        recall = recall_score(y_test, predicts)
        precision = precision_score(y_test, predicts)
        f1 = f1_score(y_test, predicts)
        auc = roc_auc_score(y_test, predicts)
        log = log_loss(y_test, predicts)

        # Registrar métricas
        mlflow.log_metric("acuracia", acuracia)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("log", log)

        # Gráficos
        # Matriz de confusão
        disp_conf = ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, cmap='Blues'
        )
        plt.title("Confusion Matrix - Random Forest")
        plt.savefig("confusionrf.png")
        plt.close()

        # Curva ROC
        disp_roc = RocCurveDisplay.from_estimator(
            model, X_test, y_test,        )
        plt.title("ROC Curve - Random Forest")
        plt.savefig("rocfr.png")
        plt.close()

        # Logar gráficos
        mlflow.log_artifact("confusionrf.png")
        mlflow.log_artifact("rocfr.png")

        # Modelo
        mlflow.sklearn.log_model(model, "Modelo_SVC")

        # Informações da execução
        print("Modelo: ", mlflow.active_run().info.run_uuid)

    mlflow.end_run()

df = read_dataset('Credit.csv')
X_train, X_test, y_train, y_test = split_dataset(df)
train_classifier(X_train, X_test, y_train, y_test)
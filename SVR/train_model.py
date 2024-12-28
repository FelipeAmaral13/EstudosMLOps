import pandas as pd
import mlflow
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

# Carregar o dataset
df = pd.read_csv('casas.csv')

# Separar features e target
X = df.drop('preco', axis=1)
y = df['preco'].copy()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Funções auxiliares
def plot_pair(df, save_path=None):
    pair_plot = sns.pairplot(df)
    fig = pair_plot.fig
    
    if save_path:
        fig.savefig(save_path, format="png", dpi=600)
    
    return fig

def count_plot(df, save_path=None):
    numeric_cols = df.select_dtypes(include=['number']).columns

    if numeric_cols.empty:
        print("Não há colunas numéricas no DataFrame.")
        return None  # Retornar None explicitamente se não houver colunas numéricas
    
    figures = []  # Lista para armazenar as figuras
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[col].dropna(), bins=10, color='blue', alpha=0.7)
        ax.set_title(f'Histograma de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequência')
        ax.grid(True)
        
        if save_path:
            plt.savefig(f"{save_path}/{col}_histogram.png", format="png", dpi=600)
        figures.append(fig)  # Adicionar a figura à lista
    
    return figures  # Retornar as figuras



def cria_experimento(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)

def callback(study, frozen_trial):
    winner = study.user_attrs.get("winner", None)
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Tentativa {frozen_trial.number} alcançou o valor: {frozen_trial.value} com "
                f"{improvement_percent: .4f}% de melhora na métrica de avaliação.")
        else:
            print(
                f"Tentativa inicial {frozen_trial.number} alcançou o valor: "
                f"{frozen_trial.value} na métrica de avaliação.")

def main():
    # Configurar o URI de rastreamento do MLflow
    mlflow.set_tracking_uri('http://127.0.0.1:8282')
    
    # Criar experimento no MLflow
    experiment_id = cria_experimento('house-prices-svr')

    # Visualizar os dados com histogramas
    with mlflow.start_run(experiment_id=experiment_id):
        # Logar pairplot
        pairplot_fig = plot_pair(df)
        if pairplot_fig:
            mlflow.log_figure(pairplot_fig, "pairplot.png")

        # Logar histogramas individuais
        count_figures = count_plot(df, save_path=None)
        if count_figures:
            for i, fig in enumerate(count_figures):
                mlflow.log_figure(fig, f"count_plot_{i}.png")
                plt.close(fig)  # Fechar a figura após salvar para liberar memória

        # Configuração de Optuna para otimização de hiperparâmetros
        def otimiza_hiperparametro(trial):
            params = {
                "C": trial.suggest_float("C", 0.1, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True)
            }

            svr = SVR(C=params["C"], kernel=params["kernel"], epsilon=params["epsilon"])
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_test)

            # Métrica de avaliação
            mse = mean_squared_error(y_test, y_pred)
            return mse  # Minimizar o erro

        # Iniciar otimização com Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(otimiza_hiperparametro, n_trials=10, callbacks=[callback])

        # Logar os melhores parâmetros e resultados no MLflow
        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_mse", best_value)

        # Treinar modelo final com os melhores parâmetros
        svr = SVR(**best_params)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)

        # Calcular métricas finais
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Logar modelo final
        artifact_path = "model"
        mlflow.sklearn.log_model(sk_model=svr,
                             artifact_path=artifact_path,
                             input_example=X_train[0:1],
                             metadata={"model_data_version": 1})
        model_uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Modelo registrado em: {model_uri}")

if __name__ == '__main__':
    main()

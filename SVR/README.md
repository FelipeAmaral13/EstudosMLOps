
# Guia Completo de Uso do Código de Regressão com SVR e MLflow

Este repositório apresenta um pipeline de aprendizado de máquina que utiliza **SVR (Support Vector Regressor)** para prever preços de casas. O projeto integra **Optuna** para otimização de hiperparâmetros e **MLflow** para rastreamento e gerenciamento de experimentos.

---

## Requisitos

Certifique-se de ter os seguintes requisitos instalados no seu ambiente:

- Python 3.7 ou superior
- Bibliotecas:
  ```bash
  pip install pandas scikit-learn mlflow optuna seaborn matplotlib
  ```

---

## Estrutura do Código

1. **Carregamento e Pré-processamento dos Dados**:
   - O dataset `casas.csv` é carregado e dividido em `features` (`X`) e `target` (`y`).
   - Divisão em treino e teste usando `train_test_split`.
   - Padronização dos dados com `StandardScaler`.

2. **Visualização de Dados**:
   - Geração de gráficos como `pairplot` e histogramas das variáveis numéricas.
   - Registro dessas visualizações no MLflow.

3. **Otimização de Hiperparâmetros**:
   - Uso do **Optuna** para otimizar os parâmetros do modelo `SVR`:
     - `C`: Regularização.
     - `kernel`: Tipo de kernel (`linear`, `rbf`, `poly`).
     - `epsilon`: Tolerância para o erro na função de perda.

4. **Treinamento do Modelo**:
   - O modelo `SVR` é treinado com os melhores hiperparâmetros encontrados.
   - Métricas como `MSE`, `RMSE` e `R²` são calculadas e registradas no MLflow.

5. **Salvamento e Carregamento do Modelo**:
   - O modelo treinado é salvo como um artefato no MLflow.
   - É possível carregar o modelo registrado para realizar previsões.

6. **Serviço de Modelo com MLflow**:
   - Servidor MLflow para gerenciar experimentos e servir o modelo treinado como uma API REST.

---

## Configuração do Servidor MLflow

### Iniciar o Servidor de Rastreamento

Para iniciar o servidor MLflow, use o comando:

```bash
mlflow server --host 127.0.0.1 --port 8282
```

O servidor será iniciado em [http://127.0.0.1:8282](http://127.0.0.1:8282).

---

## Execução do Código

1. **Edite o Dataset**:
   - Coloque o arquivo `casas.csv` no mesmo diretório do script.
   - Certifique-se de que o arquivo contém uma coluna `preco` para a variável alvo e outras colunas numéricas como features.

2. **Execute o Script**:
   - Para rodar o script principal, use:
     ```bash
     python train_model.py
     ```

   - Ele irá:
     - Carregar o dataset.
     - Dividir os dados em treino e teste.
     - Gerar visualizações e registrar no MLflow.
     - Otimizar hiperparâmetros com Optuna.
     - Treinar o modelo com os melhores parâmetros.
     - Salvar o modelo treinado no MLflow.

3. **Modelo Treinado**:
   - O modelo será registrado como um artefato no MLflow.
   - A saída mostrará o caminho absoluto onde o modelo foi salvo.

---

## Como Obter os Caminhos Absolutos

- **Local do modelo salvo**:
  Após executar o código, o caminho absoluto do modelo será exibido no console, por exemplo:
  ```
  Modelo registrado em: E:\Estudos\mlartifacts\385109406993758075\ff4a0d139fa246069fb7ad943354e670\artifacts\model
  ```

- **Dataset usado para previsão**:
  O arquivo `casas_X.csv` contém os dados para previsão.

---

## Carregar e Prever com o Modelo

### Carregamento do Modelo Treinado

Use o código abaixo para carregar o modelo e realizar previsões em novos dados:

```python
import mlflow
import pandas as pd

# Caminho do modelo salvo
logged_model = r'E:\Estudos\mlartifacts\385109406993758075\ff4a0d139fa246069fb7ad943354e670\artifacts\model'

# Carregar o modelo como um PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Carregar os dados para previsão
data = pd.read_csv('casas_X.csv')
predicted = loaded_model.predict(pd.DataFrame(data))

# Salvar as previsões em um arquivo CSV
data['predicted'] = predicted
data.to_csv('precos.csv')
```

---

## Servir o Modelo como API REST

Você pode expor o modelo treinado como uma API usando o MLflow:

### Comando para Servir o Modelo

```bash
mlflow models serve -m "E:\Estudos\mlartifacts\385109406993758075\ff4a0d139fa246069fb7ad943354e670\artifacts\model" -p 8283 --no-conda
```

### Realizar Previsões com a API

1. Use ferramentas como `curl` ou bibliotecas como `requests` no Python.
2. Exemplo de solicitação com `curl`:

```bash
curl -X POST -H "Content-Type: application/json" --data '{"columns":["feature1", "feature2"], "data":[[value1, value2]]}' http://127.0.0.1:8283/invocations
```

---

## Métricas Registradas no MLflow

As seguintes métricas serão registradas no MLflow:
- `mse` (Erro Médio Quadrado)
- `rmse` (Raiz do Erro Médio Quadrado)
- `r2` (Coeficiente de Determinação)
- `best_mse` (Melhor MSE encontrado pela otimização)

Você pode visualizar essas métricas no painel do MLflow acessando [http://127.0.0.1:8282](http://127.0.0.1:8282).

---

## Estrutura Esperada de Arquivos

```
project/
│
├── casas.csv               # Dataset para treino
├── casas_X.csv             # Dados para previsão
├── train_model.py          # Script principal
├── precos.csv              # Saída com as previsões
├── mlartifacts/            # Diretório gerado pelo MLflow
```

---

## Considerações Finais

- Certifique-se de que os caminhos dos arquivos estão corretos ao rodar os comandos.
- Utilize Optuna para ajustar o número de tentativas e melhorar a busca por hiperparâmetros.
- Caso tenha dúvidas, revise os logs do MLflow e os arquivos gerados para entender os passos executados.

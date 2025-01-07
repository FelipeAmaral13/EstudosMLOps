
# Implementação de Modelo de Machine Learning com MLflow

## Descrição do Projeto
Este projeto implementa um modelo de classificação utilizando o algoritmo **SVC (Support Vector Classifier)** da biblioteca Scikit-learn. O experimento é gerenciado e registrado no **MLflow**, uma ferramenta para controle de experimentos, rastreamento de métricas, e armazenamento de artefatos como gráficos e modelos treinados. O dataset utilizado é o `Credit.csv`, que deve conter informações relevantes para classificação.

## Estrutura do Projeto
O projeto possui as seguintes funções principais:

1. **read_dataset:**
   - Lê o dataset em formato CSV e transforma colunas categóricas em numéricas.

2. **split_dataset:**
   - Realiza a separação do dataset em conjuntos de treino e teste.

3. **train_classifier:**
   - Treina o modelo de classificação SVC, registra métricas, e salva gráficos e o modelo no MLflow.

## Tecnologias Utilizadas
- **Python 3.8+**
- **Pandas** para manipulação de dados.
- **Scikit-learn** para treinamento do modelo e cálculo de métricas.
- **Matplotlib** para visualização.
- **MLflow** para rastreamento e gerência dos experimentos.

## Configuração do Ambiente
1. Clone este repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <PASTA_DO_REPOSITORIO>
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. Execute o servidor MLflow:
   ```bash
   mlflow ui
   ```
   O servidor estará acessível em [http://localhost:5000](http://localhost:5000).

## Execução do Projeto
1. Certifique-se de que o arquivo `Credit.csv` está no diretório do projeto.
2. Execute o script principal:
   ```bash
   python main.py
   ```
3. O treinamento do modelo será iniciado, e os resultados serão registrados no MLflow.

## Métricas Calculadas
- **Acurácia**
- **Recall**
- **Precisão**
- **F1 Score**
- **AUC (Area Under the Curve)**
- **Log Loss**

## Artefatos Gerados
1. **Gráficos:**
   - Matriz de Confusão (`confusionrf.png`)
   - Curva ROC (`rocfr.png`)

2. **Modelo:**
   - Modelo treinado `Modelo_SVC` salvo no MLflow.

## Estrutura do Código
### Funções Principais
#### 1. `read_dataset(path: str)`
Carrega o dataset, realiza a codificação de variáveis categóricas para numéricas e retorna um DataFrame do Pandas.

#### 2. `split_dataset(df: pd.DataFrame)`
Divide o dataset em conjuntos de treino e teste com proporção de 70/30.

#### 3. `train_classifier(X_train, X_test, y_train, y_test)`
- Configura o experimento no MLflow.
- Treina o modelo **SVC**.
- Calcula e registra as métricas no MLflow.
- Gera e salva gráficos (matriz de confusão e curva ROC).
- Salva o modelo no MLflow.

### Implementação de Modelo no MLflow
O MLflow é utilizado para:
1. **Registrar métricas:**
   ```python
   mlflow.log_metric("acuracia", acuracia)
   mlflow.log_metric("recall", recall)
   mlflow.log_metric("precision", precision)
   mlflow.log_metric("f1", f1)
   mlflow.log_metric("auc", auc)
   mlflow.log_metric("log", log)
   ```

2. **Logar gráficos como artefatos:**
   ```python
   mlflow.log_artifact("confusionrf.png")
   mlflow.log_artifact("rocfr.png")
   ```

3. **Salvar o modelo treinado:**
   ```python
   mlflow.sklearn.log_model(model, "Modelo_SVC")
   ```

4. **Definir tags para descrição do experimento:**
   ```python
   mlflow.set_tag("model", "Support Vector Classifier (SVC)")
   mlflow.set_tag("author", "Felipe Amaral")
   mlflow.set_tag("objective", "Avaliar performance inicial do modelo SVC para o dataset Credit")
   ```

## Uso do Modelo como Serviço no MLflow
Após salvar o modelo no MLflow, é possível utilizá-lo como um serviço para previsões. Para isso, siga os passos abaixo:

1. **Inicie o Serviço do Modelo:**
   Localize o ID do modelo salvo no MLflow UI. Em seguida, execute:
   ```bash
   mlflow models serve -m "runs:/<RUN_ID>/Modelo_SVC" -p 1234
   ```
   Isso iniciará um servidor RESTful na porta 1234.

2. **Envie uma Requisição para o Modelo:**
   Use uma ferramenta como `curl` ou qualquer biblioteca HTTP para enviar uma requisição POST com os dados no formato JSON. Exemplo:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"instances": [[0.5, 1.2, 3.4, ...]]}' \
        http://127.0.0.1:1234/invocations
   ```

3. **Receba a Resposta:**
   O modelo retornará as previsões no formato JSON:
   ```json
   {
       "predictions": [1]
   }
   ```

4. **Integração com Aplicativos:**
   Você pode integrar o endpoint do modelo com aplicações web, dashboards ou scripts de automação para previsões em tempo real.

## Resultados Esperados
Ao final da execução, você poderá:
- Visualizar as métricas do modelo diretamente no MLflow.
- Obter os gráficos da matriz de confusão e da curva ROC.
- Acessar o modelo treinado e utilizá-lo em aplicações futuras.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença
Este projeto está licenciado sob a [MIT License](LICENSE).

---

**Autor:** Felipe Amaral

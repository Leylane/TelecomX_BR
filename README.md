# 📊 Telecom X — Análise de Evasão de Clientes (Churn)

Este projeto faz parte do desafio de **Data Science aplicada a Churn Prediction**, onde analisamos os fatores que levam clientes da **Telecom X** a cancelar seus serviços.  
A análise percorre **ETL (Extração/Transformação)** → **EDA (Análise Exploratória)** → **Modelagem Preditiva** → **Recomendações**.

---

## 🚀 Objetivo

- Entender os **padrões** associados ao churn.  
- Construir **baselines preditivos** (Logistic Regression e Random Forest).  
- Gerar **insights acionáveis** para reduzir a evasão e aumentar retenção/LTV.

---

## 🗂️ Estrutura do Projeto

```text
TelecomX_BR/
│── dataset/
│   └── TelecomX_Data.json        # Fonte de dados (lido via RAW)
│
│── notebooks/
│   └── TelecomX_Churn_End_to_End.ipynb   # Notebook principal
│
│── src/
│   ├── etl.py       # Funções de ETL (extração/limpeza/feature engineering)
│   ├── eda.py       # Funções de EDA (tabelas e gráficos Matplotlib)
│   ├── models.py    # Baselines de modelagem (LR e RF)
│   └── utils.py     # Helpers (seed, leitura RAW, formatação de métricas)
│
│── README.md
│── requirements.txt
```

---

## 📥 Fonte de Dados

- **RAW**: [TelecomX_Data.json](https://raw.githubusercontent.com/Leylane/TelecomX_BR/refs/heads/main/dataset/TelecomX_Data.json)  
- O notebook lê **diretamente** do link acima (sem salvar arquivos locais).

Exemplo de leitura (trecho usado no notebook):

```python
import json, requests, pandas as pd

RAW_URL = "https://raw.githubusercontent.com/Leylane/TelecomX_BR/refs/heads/main/dataset/TelecomX_Data.json"

def load_json_any_text(txt: str):
    txt = txt.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # Suporte a JSON Lines (JSONL)
        return [json.loads(line) for line in txt.splitlines() if line.strip()]

res = requests.get(RAW_URL, timeout=60)
res.raise_for_status()
raw_data = load_json_any_text(res.text)

df_raw = pd.json_normalize(raw_data, sep="_")
df_raw.head()
```

---

## 🛠️ Tecnologias

- **Python** 3.11+  
- `pandas`, `numpy` – manipulação/ETL  
- `matplotlib` – gráficos (sem temas/cores custom)  
- `scikit-learn` – modelos (LR/RF)  
- `requests` – ingestão via RAW

### 📦 Requisitos (requirements.txt)

```txt
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
requests>=2.31
```

---

## ▶️ Como Executar

### A) Online (Colab/Kaggle/VSCode Web)
1. Abra `notebooks/TelecomX_Churn_End_to_End.ipynb`.  
2. Execute célula a célula (o caderno já lê os dados do RAW).

### B) Local (Jupyter)
```bash
git clone https://github.com/Leylane/TelecomX_BR.git
cd TelecomX_BR

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/TelecomX_Churn_End_to_End.ipynb
```

---

## 🔎 Metodologia (resumo)

1. **ETL**  
   - Conversões numéricas (`MonthlyCharges`, `TotalCharges`, `tenure`, `SeniorCitizen`).  
   - Normalização de categorias (“No internet service” → “No”).  
   - *Features* derivadas: `HasInternet`, `HasPhone`, `tenure_bin`.  

2. **EDA**  
   - Tabelas de taxa de churn por: contrato, método de pagamento, tipo de internet, tenure.  
   - Gráficos de barras (Matplotlib) e histograma de `MonthlyCharges` por churn.

3. **Modelagem**  
   - Split estratificado 80/20.  
   - Imputação simples (numérico = mediana; categórico = `"Unknown"`).  
   - **Logistic Regression (liblinear)** e **Random Forest** (class_weight balanceado).  
   - Métricas: ROC_AUC, Accuracy, Precision, Recall.  
   - *Top 10* coeficientes (LR) e importâncias (RF).

---

## 📊 Resultados (resumo do teste 20%)

| Modelo              | ROC_AUC | Accuracy | Precision | Recall |
|---------------------|:-------:|:--------:|:---------:|:------:|
| Logistic Regression | **0.846** | 0.795    | 0.648     | 0.497  |
| Random Forest       | 0.823   | 0.785    | 0.631     | 0.457  |

**Padrões chave**:
- Maior risco em **contrato *Month-to-month*** e pagamento via **Electronic check**.  
- **Fiber optic** apresenta churn mais alto que DSL/sem internet.  
- **Tenure baixo** (0–6 meses) é o período mais crítico.

---

## 🧩 Como reutilizar os módulos (`src/`)

```python
from src.utils import set_seed
from src import etl, eda, models

set_seed(42)

# 1) ETL
df_valid, df_clean = etl.get_dataset()

# 2) EDA
tabs = eda.summary_tables(df_valid)
overall = eda.churn_overall(df_valid)
display(tabs.get("account_Contract"))
figs = eda.plot_key_charts(tabs)  # retorna dict de figuras

# 3) Modelagem
metrics_df, top_lr_df, top_rf_df = models.run_baselines(df_clean)
display(metrics_df)
display(top_lr_df)
display(top_rf_df)
```

---

## 📌 Recomendações (anti-churn)

1. **Migrar contratos mensais para anuais** com benefício nos primeiros 90 dias.  
2. **Incentivar troca de meio de pagamento** (*Electronic check* → débito automático/cartão).  
3. **Onboarding estendido (0–90 dias)** com suporte proativo e comunicação ativa.  
4. **Bundles de valor** (TechSupport/OnlineSecurity) para elevar valor percebido.  
5. **Priorizar retenção por propensão de churn** (scores dos modelos).

---

## 🧭 Roadmap

- Feature engineering adicional (NPS, chamados de suporte, falhas técnicas, descontos).  
- Modelos avançados (XGBoost/LightGBM) e calibração de probabilidades.  
- Pipeline MLOps com monitoramento de drift e re-treino periódico.

---

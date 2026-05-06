# Análise Exploratória de Dados — House Prices

## Equipe
* **Integrante 1**: Rafaeel Antônio da Silva Neto - 2212378
* **Integrante 2**: Emanuel Sales Marinho Rocha - 2413961
* **Integrante 3**: Marcos Aurelio de Araujo Dias - 2010430
* **Integrante 4**:  Siwan Eden Oliveira - 2220291

---

## Objetivo
Construir um modelo preditivo para estimar o preço de venda de imóveis residenciais em Ames, Iowa, a partir de 79 variáveis explicativas. A métrica de otimização principal é o **RMSLE** (Root Mean Squared Log Error), que penaliza erros proporcionais ao valor do imóvel. Este repositório é focadao para análise e tratamento dos dados.

---

## Organização do Repositório (Arquitetura)

### 1. `data/` — Dados

* **`treino.csv/`**: Dataset original com as 79 variáveis explicativas e a variável alvo `SalePrice`. **Nunca deve ser alterado.**
* **`data_description.txt/`**: Dicionário de dados detalhado. Consulte antes de tratar qualquer coluna.

### 2. `notebooks/` — Análise e Experimentação

* **`01_eda.ipynb.ipynb`**:  Análise Exploratória de Dados.

---

## Tecnologias Utilizadas

* **Linguagem**: Python 3.10+
* **Análise de Dados**: Pandas, NumPy

---

## Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

### 2. Crie e ative um ambiente virtual (recomendado)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### Configuração do Kernel no VS Code (Importante)
Após instalar as dependências no venv, você precisa garantir que o Jupyter Notebook está usando o ambiente correto:

- Abra qualquer arquivo .ipynb no VS Code.

- No canto superior direito, clique onde aparece a versão do Python (ex: Python 3.13.x).

- Selecione a opção "Python Environments...".

- Escolha o interpretador que está dentro da pasta do projeto (geralmente marcado como 'venv': venv).

- Se o VS Code pedir para instalar o ipykernel, clique em Install.

Dica: Se você não fizer isso, o código não encontrará as bibliotecas instaladas (como o pandas ou nltk), mesmo que o terminal diga que está tudo certo.


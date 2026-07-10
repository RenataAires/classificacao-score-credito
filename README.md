# 📊 Sistema de Classificação de Score de Crédito de Clientes

Este repositório contém o pipeline completo de Engenharia de Dados e Machine Learning desenvolvido para a **Data Girls Finance**. O objetivo principal é automatizar, auditar e classificar o perfil de risco de crédito de clientes em três categorias: **Poor (0)**, **Standard (1)** e **Good (2)**, mitigando taxas de inadimplência crônica e otimizando a concessão de limites de forma ética e escalável.

---

## 🛠️ Tecnologias e ferramentas utilizadas
* **Linguagem Principal:** Python 3.12
* **Manipulação e Engenharia de Dados:** Pandas, NumPy
* **Modelagem Preditiva & Métricas:** Scikit-Learn
* **Inteligência Artificial Explicável (XAI):** SHAP (SHapley Additive exPlanations)
* **Visualização de Dados:** Matplotlib, Seaborn

---

## 🚀 1. Ciclo de vida do pipeline

## 📖 Dicionário de Dados (Data Dictionary)

Abaixo estão descritas todas as variáveis utilizadas na modelagem preditiva final, seus tipos de dados após o pipeline de engenharia e suas respectivas definições de negócio:

| Variável | Tipo de Dado | Descrição de Negócio |
| :--- | :--- | :--- |
| `Age` | Inteiro (`int64`) | Idade do cliente no momento da análise. |
| `Annual_Income` | Float (`float64`) | Renda anual bruta declarada e validada do cliente. |
| `Monthly_Inhand_Salary` | Float (`float64`) | Salário líquido estimado que o cliente efetivamente recebe por mês. |
| `Num_Bank_Accounts` | Inteiro (`int64`) | Quantidade de contas bancárias ativas em nome do cliente. |
| `Num_Credit_Card` | Inteiro (`int64`) | Quantidade de cartões de crédito que o cliente possui. |
| `Interest_Rate` | Inteiro (`int64`) | Taxa de juros média cobrada nas linhas de crédito ativas do cliente. |
| `Num_of_Loan` | Inteiro (`int64`) | Quantidade de empréstimos ativos contratados pelo cliente. |
| `Delay_from_due_date` | Inteiro (`int64`) | Média de dias de atraso do cliente em relação à data de vencimento da fatura. |
| `Num_of_Delayed_Payment` | Float (`float64`) | Quantidade total de pagamentos que o cliente já atrasou no histórico. |
| `Num_Credit_Inquiries` | Float (`float64`) | Número de consultas de score de crédito realizadas por instituições nas últimas semanas. |
| `Credit_Mix` | Inteiro (`int64`) | Qualidade e variedade do mix de crédito do cliente ($0$: Ruim, $1$: Regular, $2$: Bom). |
| `Outstanding_Debt` | Float (`float64`) | Volume total de dívida pendente e em aberto acumulada pelo cliente. |
| `Credit_Utilization_Ratio` | Float (`float64`) | Taxa de utilização do limite de crédito disponível (quanto ele gasta do total que tem). |
| `Amount_invested_monthly` | Float (`float64`) | Valor financeiro que o cliente poupa/investe mensalmente. |
| `Payment_of_Min_Amount` | Inteiro (`int64`) | Sinalizador se o cliente costuma pagar apenas o mínimo da fatura ($0$: Não, $1$: Sim). |
| `Total_EMI_per_month` | Float (`float64`) | Valor total pago mensalmente em parcelas de empréstimos/financiamentos (Equated Monthly Installment). |
| `Monthly_Balance` | Float (`float64`) | Saldo financeiro que sobra na conta do cliente ao final do mês. |
| `Pay_v_High_spent_Large_value_payments` | Binário (`int64`) | Flag ($0$ ou $1$): Indica alto volume de gastos e pagamentos de altos valores. |
| `Pay_v_High_spent_Medium_value_payments` | Binário (`int64`) | Flag ($0$ ou $1$): Indica alto volume de gastos e pagamentos de médios valores. |
| `Pay_v_High_spent_Small_value_payments` | Binário (`int64`) | Flag ($0$ ou $1$): Indica alto volume de gastos e pagamentos de pequenos valores. |
| `Pay_v_Low_spent_Large_value_payments` | Binário (`int64`) | Flag ($0$ ou $1$): Indica baixo volume de gastos e pagamentos de altos valores. |
| `Pay_v_Low_spent_Medium_value_payments` | Binário (`int64`) | Flag ($0$ ou $1$): Indica baixo volume de gastos e pagamentos de médios valores. |
| `Pay_v_Low_spent_Small_value_payments` | Binário (`int64`) | Flag ($0$ ou $1$): Indica baixo volume de gastos e pagamentos de pequenos valores. |
| `Credit_Score` | **Target** (`int64`) | Variável Alvo. Classificação final do risco de crédito ($0$: Poor/Ruim, $1$: Standard/Médio, $2$: Good/Excelente). |

### Etapa 1: Leitura e Limpeza Profunda (Data Cleaning)
A base bruta continha anomalias severas vindas do sistema legado que inviabilizariam qualquer modelo preditivo. As ações de engenharia incluíram:
* **Remoção de Ruídos de String:** Identificação e limpeza de caracteres especiais ocultos (`_`) e siglas órfãs (`NM`, `!@9#%8`).
* **Tratamento de Outliers Críticos:** A coluna `Monthly_Balance` apresentava corrupção de dados com valores na casa dos septilhões negativos ($-3.33 \times 10^{26}$). O problema foi mitigado aplicando a substituição pela mediana dos saldos reais.
* **Dados Faltantes:** Linhas corrompidas e nulas foram saneadas utilizando a moda e a mediana estatística da distribuição da base.

### Etapa 2: Engenharia de Recursos (Feature Engineering)
Para traduzir o comportamento de negócio para a matemática dos algoritmos, foram aplicadas técnicas distintas de codificação:
1. **Mapeamento Ordinal (Hierárquico):** Colunas como `Credit_Mix` e o nosso alvo `Credit_Score` foram mapeadas manualmente ($0$, $1$ e $2$), preservando a ordem lógica de valor financeiro.
2. **Mapeamento Binário:** A coluna `Payment_of_Min_Amount` foi convertida em sinalizadores numéricos ($0$ e $1$).
3. **One-Hot Encoding:** Variáveis categóricas nominais e sem hierarquia clara, como `Payment_Behaviour`, foram desmembradas em 6 novas colunas binárias (`Pay_v_...`), garantindo que o modelo aprenda o peso isolado de cada comportamento sem criar vieses artificiais de magnitude.

---

## 📉 2. Modelagem preditiva e resultados

O dataset de 100.000 linhas foi dividido utilizando a estratégia de validação cruzada holding: **80% para Treino** (80.000 linhas) e **20% para Teste** (20.000 linhas), usando uma semente aleatória reproduzível (`random_state=42`).

| Modelo Avaliado | Condição / Ajuste | Acurácia Alcançada |
| :--- | :--- | :--- |
| **Decision Tree (Baseline)** | `max_depth=10` | **69.89%** |
| **Random Forest (Padrão)** | Configurações Base | **72.19%** |
| **Random Forest (Otimizado)** | `GridSearchCV` (150 estimadores, profundidade 15) | **74.09%** |

O ajuste fino (*Hyperparameter Tuning*) via `GridSearchCV` permitiu extrair quase **2% de acurácia incremental** apenas calbrando o comportamento interno das árvores do ensemble, consolidando o modelo final com **74.09% de acerto global**.

---

## 🔍 3. Avaliação de impacto de negócio e diagnóstico de erros

A análise fria da acurácia global não basta. Foi avaliado a performance sob a ótica da **Matriz de Confusão**:

* **Falsos Negativos de Risco (1.465 casos):** Clientes de perfil real *Poor* que o modelo classificou equivocadamente como *Standard* ou *Good*. Este representa o **erro mais caro para a fintech**, pois resulta em concessão indevida de crédito e potencial aumento do NPL (*Non-Performing Loans* / Crédito Podre).
* **Falsos Positivos de Risco (1.780 casos):** Clientes saudáveis (*Good*/*Standard*) classificados como *Poor*. Representam um **custo de oportunidade**, onde a empresa deixa de faturar receita de juros e tarifas para evitar um risco inexistente.
* **Segurança Operacional:** O modelo demonstrou alta confiabilidade extrema, confundindo apenas **83** clientes excelentes (`Good`) diretamente com a categoria `Poor`.

### 🧠 Importância das variáveis e explicabilidade SHAP
Ao extrair os pesos globais do Random Forest e cruzar com o impacto local gerado pela biblioteca **SHAP**, foram mapeados os fatores determinantes que empurram o score de um cliente para o nível **Poor**:

1. **Outstanding_Debt (Dívida Pendente):** O maior peso isolado. Clientes com alto volume de endividamento ativo têm chances exponencialmente maiores de rebaixamento de score.
2. **Interest_Rate (Taxa de Juros):** Estar atrelado a linhas de financiamento caras deteriora a capacidade de pagamento futura.
3. **Delay_from_due_date (Dias de Atraso):** A frequência diária com que o cliente deixa passar o vencimento das faturas.
4. **Num_Credit_Inquiries (Consultas de Crédito):** O comportamento de buscar múltiplos créditos no mercado de forma simultânea dispara o gatilho de risco do algoritmo.

---

## 💡 4. Respostas às perguntas norteadoras de negócio

* **Quais variáveis podem ter maior relação com a capacidade de pagamento dos clientes?**
  Dívida total acumulada (`Outstanding_Debt`), taxa de juros contratada (`Interest_Rate`) e a quantidade de dias que o cliente atrasa após o vencimento (`Delay_from_due_date`).
* **Qual perfil de cliente possui maior probabilidade de ser classificado com score “Poor”?**
  Clientes com alto endividamento ativo, histórico de múltiplos cartões de crédito abertos, alta recorrência de pagamento apenas do valor mínimo da fatura e histórico frequente de atrasos diários superiores a duas semanas.
* **Um modelo de Machine Learning consegue classificar o score de crédito com boa performance?**
  Sim. A baseline de árvore entregou quase 70% e o Random Forest tunado alcançou **74.09%**. Para dados reais altamente ruidosos, este é um patamar robusto e produtivo para iniciar uma operação de crédito assistida.
* **Como os resultados do modelo poderiam apoiar uma equipe de crédito na prática?**
  Criando esteiras automatizadas: aprovação automática e aumento de limite instantâneo para previsões `Good` com confiança $>90\%$; congelamento preventivo de novos créditos para perfis marcados como `Poor`; e direcionamento de réguas de cobrança preventiva para clientes na zona cinzenta do perfil `Standard`.

---

## ⚖️ 5. Governança e cuidados éticos em produção

Antes de disponibilizar este modelo em uma API de produção, a engenharia e a governança da *Data Girls Finance* devem garantir:
1. **Direito à Explicabilidade (LGPD & CDC):** Modelos *Black-Box* (como Random Forest) não podem simplesmente negar crédito sem justificativa. Devemos usar a saída matemática do SHAP para automatizar mensagens claras e transparentes ao cliente do motivo exato da negativa (*"Seu limite não pôde ser expandido no momento devido ao volume de dívidas pendentes associadas"*).
2. **Monitoramento de Data Drift:** O comportamento do mercado financeiro e a inflação mudam. O modelo treinado hoje ficará obsoleto se as variáveis macroeconômicas mudarem. Recomenda-se um pipeline automatizado de monitoramento para retreinamento programado a cada **6 meses**.
3. **Auditoria contra Vieses (Bias Fairness):** Garantir que o modelo não utilize de forma indireta marcadores demográficos, geográficos ou profissionais para discriminar ou penalizar grupos de clientes de forma sistêmica e injusta.

---

## 📂 Como Executar este Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/RenataAires/classificacao-score-credito.git
   ```
---
2. Instale as dependências necessárias:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib shap
   ```
---

3. Abra o Jupyter Notebook ou envie o arquivo .ipynb diretamente para o Google Colab para executar o pipeline de ponta a ponta.

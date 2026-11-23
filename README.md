# Resíduos Sólidos de Fortaleza: que resíduo é esse e quanto ele pesa?

Projeto da disciplina de Ciência de Dados (UNIFOR). O objetivo é classificar o tipo de
resíduo de cada pesagem de caminhão no aterro de Caucaia (ASMOC), que recebe o lixo de
Fortaleza e Caucaia, usando só os dados da balança, do caminhão, da empresa e do lugar de
coleta. Como segunda tarefa, estimamos por regressão o peso líquido de cada carga e
comparamos qual das duas os dados permitem resolver melhor.

Equipe: Ana Beatriz de Sousa Colaço, Gabriel Pereira Albuquerque, Thiago Cavalcante da
Silva e Leonardo Coelho Martins.

## Dados

Portal de Dados Abertos da Prefeitura de Fortaleza, arquivos `rs-2022.csv`, `rs-2023.csv`
e `rs-2024.csv` (788.525 pesagens no total). Os arquivos não estão no repositório: baixe e
coloque em `data/raw/`. O código aceita tanto o formato do portal (ponto e vírgula, CP1252)
quanto o de um arquivo salvo de novo numa planilha (vírgula, UTF-8).

## Como rodar

```bash
pip install -r requirements.txt
cd src
python sprint1_eda.py              # Sprint 1: análise exploratória
python sprint2_preprocessamento.py # Sprint 2: limpeza, treino/teste, codificação
python sprint3_modelos.py          # Sprint 3: algoritmos individuais
python sprint4_comites.py          # Sprint 4: comitês homogêneos e heterogêneos
python sprint5_regressao.py        # Regressão do peso líquido e comparação das tarefas
```

Cada script grava tabelas em `reports/tables/` e figuras em `reports/figures/<sprint>/`.
A Sprint 4 depende dos arquivos gerados pelas Sprints 2 e 3. A regressão usa as bases
`regressao_treino.csv.gz` e `regressao_teste.csv.gz` da Sprint 2, que têm exatamente as
mesmas linhas de treino e teste da classificação; a comparação entre as tarefas usa
também o `s4_metricas.csv`.

## Estrutura

```
src/
  config.py                   caminhos, classes, cores, semente
  carregar.py                 leitura dos 3 CSVs
  preprocessamento.py         funções de limpeza e codificação
  graficos.py                 estilo único dos gráficos
  sprint1_eda.py
  sprint2_preprocessamento.py
  sprint3_modelos.py
  sprint4_comites.py
  sprint5_regressao.py        regressão do peso líquido
reports/
  tables/   s1_* … s5_*
  figures/  sprint1/ … sprint5/
```

## Resumo por sprint

| Sprint | Entrega | Resultado principal |
| --- | --- | --- |
| 1 | Análise exploratória | 6 classes desbalanceadas (Domiciliar 34,2%, Varrição 1,9%); duplicatas e codificação quebrada em 2024 |
| 2 | Pré-processamento | 710.515 pesagens e 96 atributos; divisão 70/30 estratificada, semente 42 |
| 3 | Algoritmos individuais | Árvore (F1 macro 0,899) e rede neural (0,893) lideram no teste |
| 4 | Comitês | Bagging, Random Forest, AdaBoost, Gradient Boosting, votação e stacking |
| Regressão | Peso líquido (t) | Random Forest: MAE 1,58 t e R² 0,81, contra 3,85 t de quem chuta a média |

Comparando as tarefas pela redução do erro em relação à referência de cada uma, a
classificação reduziu o erro em 94% e a regressão em 59%
(`reports/tables/s5_comparacao_tarefas.csv`).

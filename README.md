# Resíduos Sólidos de Fortaleza: que resíduo é esse?

Projeto da disciplina de Ciência de Dados (UNIFOR). O objetivo é classificar o tipo de
resíduo de cada pesagem de caminhão no aterro de Caucaia (ASMOC), que recebe o lixo de
Fortaleza e Caucaia, usando só os dados da balança, do caminhão, da empresa e do lugar de
coleta.

Equipe: Ana Beatriz de Sousa Colaço, Gabriel Pereira Albuquerque, Thiago Cavalcante da
Silva e Leonardo Coelho Martins.

## Dados

Portal de Dados Abertos da Prefeitura de Fortaleza, arquivos `rs-2022.csv`, `rs-2023.csv`
e `rs-2024.csv` (788.525 pesagens no total). Os arquivos não estão no repositório: baixe e
coloque em `data/raw/`.

## Como rodar

```bash
pip install -r requirements.txt
cd src
python sprint1_eda.py              # Sprint 1: análise exploratória
python sprint2_preprocessamento.py # Sprint 2: limpeza, treino/teste, codificação
python sprint3_modelos.py          # Sprint 3: algoritmos individuais
python sprint4_comites.py          # Sprint 4: comitês homogêneos e heterogêneos
```

Cada script grava tabelas em `reports/tables/` e figuras em `reports/figures/<sprint>/`.
A Sprint 4 depende dos arquivos gerados pelas Sprints 2 e 3.

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
reports/
  tables/   s1_* … s4_*
  figures/  sprint1/ … sprint4/
```

## Resumo por sprint

| Sprint | Entrega | Resultado principal |
| --- | --- | --- |
| 1 | Análise exploratória | 6 classes desbalanceadas (Domiciliar 34,2%, Varrição 1,9%); duplicatas e codificação quebrada em 2024 |
| 2 | Pré-processamento | 710.515 pesagens e 96 atributos; divisão 70/30 estratificada, semente 42 |
| 3 | Algoritmos individuais | Árvore (F1 macro 0,899) e rede neural (0,893) lideram no teste |
| 4 | Comitês | Bagging, Random Forest, AdaBoost, Gradient Boosting, votação e stacking |

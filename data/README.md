# Dataset

Arquivo usado: `HPLC_data.csv`, com 13.031 linhas e 19 colunas. Alvo: `Diagnosis`.

Origem informada no Trabalho 1: [HPLC-Based Thalassemia Screening Data — Kaggle](https://www.kaggle.com/datasets/abhraghoshcmc/hplc-based-thalassemia-screening-data).

O CSV `HPLC_data.csv` é o mesmo do Trabalho 1 e está versionado nesta pasta. Licença dos dados: **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/) e sob a mesma licença.

## Partição oficial

`splits/split_indices.csv` guarda a partição 60/20/20 estratificada (semente 42) do Trabalho 1, após remover as 5 classes sem suporte e os 8 registros implausíveis (13.011 linhas, 8 classes): treino 7.806, validação 2.602 e teste 2.603. Colunas: `row_index` (índice da linha no CSV), `sl_no`, `subset` (`train`, `val`, `test`) e `diagnosis`. O arquivo é gerado pelo `notebooks/01_pipeline_manual.ipynb`, e `src/c318_common.load_split()` o confere por asserção.

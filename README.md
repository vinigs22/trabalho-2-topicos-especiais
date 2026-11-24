# Trabalho 2 — Aprendizado de Máquina Automatizado

Disciplina C318 — Tópicos Especiais II, Inatel. Equipe 4.

Integrantes: Gabriel Lopes Silva, Lucas Caetano Reis, Pedro Tiso Vinhas Mesquita e Vinicius Gonçalves Simoni.

## Etapa atual

Este repositório registra os testes preliminares realizados no Google Colab em 12/11/2026 para a ficha de acompanhamento. As ferramentas escolhidas são **TPOT 1.1.0** (busca sobre pipelines scikit-learn) e **AutoGluon Tabular 1.6.3** (família de gradient boosting automatizado na tabela do enunciado).

Os testes confirmaram instalação, importação e treinamento no CSV. **Ainda não constituem a comparação oficial com o Trabalho 1.** O notebook foi reconstruído a partir das células e saídas compartilhadas pela equipe; não é uma exportação do caderno original e não contém saídas de execução fabricadas.

## Arquivos

- `notebooks/01_pipeline_manual.ipynb`: **protocolo oficial** — dados e filtros do Trabalho 1, partição 60/20/20 fixa, pipeline manual de referência (T1 corrigido), baseline de classe majoritária, métricas, tempo, memória e importância por permutação. Resultados em `resultados/manual/`.
- `src/c318_common.py`: funções compartilhadas por todos os notebooks oficiais (leitura do CSV, filtros do T1, partição, asserções, métricas e medição de tempo/memória).
- `notebooks/02_tpot_oficial.ipynb`: experimento oficial do **TPOT**. Executa a curva de budgets, seleciona o budget exclusivamente pela validação, realiza a ablação dos atributos derivados, calcula importância por permutação e reserva o teste para uma única consulta final.
- `src/tpot_protocol.py`: prepara a entrada do TPOT e concentra a configuração, a busca e a auditoria da ferramenta.
- `data/HPLC_data.csv` e `data/splits/split_indices.csv`: CSV do Trabalho 1 e índices de treino/validação/teste usados por todos os sistemas.
- `requirements-manual.txt`: versões exatas do pipeline manual.
- `notebooks/testes_preliminares_colab.ipynb`: instalação, preparação e testes das duas ferramentas, com os ajustes de API usados no Colab.
- `data/README.md`: origem e localização do CSV.
- `resultados/testes_2026-11-12.md`: resultados relatados, falhas e limitações.
- `requirements-autogluon.txt` e `requirements-tpot.txt`: versões das ferramentas testadas; não representam um congelamento completo do ambiente.

## Como executar

### Pipeline manual (notebook oficial)

```bash
git clone <url-do-repositorio>
pip install -r requirements-manual.txt
jupyter lab   # abrir notebooks/01_pipeline_manual.ipynb > Restart Kernel and Run All
```

Com `RUN_LONG = False` (padrão), os resultados salvos em `resultados/manual/` são carregados; com `True`, refeitos. O teste reservado é consultado uma única vez e o resultado fica salvo.

### Dados nos outros notebooks oficiais

Todos os sistemas usam a mesma partição. Nos notebooks das ferramentas:

```python
import sys; sys.path.insert(0, "../src")
from c318_common import *
ROOT = find_repo_root()
train_df, val_df, test_df = load_split(ROOT)   # confere os índices por asserção
```

### TPOT (notebook oficial)

1. No Google Colab, clone ou envie o repositório e execute `pip install -r requirements-tpot.txt`.
2. Abra `notebooks/02_tpot_oficial.ipynb` a partir da raiz do repositório. A primeira execução deve manter todas as chaves `RUN_*` como `False`: ela apenas confere ambiente, CSV, filtros, partição e asserções.
3. Para as 15 execuções da curva, altere somente `RUN_VALIDATION = True`. O notebook ajusta cada busca exclusivamente no treino e mede a validação externa para os budgets de 5, 30 e 60 minutos e sementes 42, 101, 223, 347 e 409.
4. Após a escolha gravada em `resultados/tpot/budget_selecionado.json`, execute separadamente `RUN_DERIVED_ABLATION = True` e `RUN_IMPORTANCE = True`.
5. Somente quando toda a seleção estiver congelada, altere `RUN_FINAL_TEST = True`. A célula impede nova consulta se `resultados/tpot/metricas_teste.csv` já existir.

O TPOT recebe apenas transformações determinísticas por registro antes da busca. O notebook usa o espaço oficial `linear-light` do TPOT 1.1.0, pois o espaço personalizado com `ColumnOneHotEncoder` apresentou erro interno na seleção da população nessa versão. Essa escolha fica registrada para auditoria; a configuração efetiva deve ser conferida na saída da pipeline e no relatório.

### Testes preliminares (ficha de acompanhamento)

1. Abra `notebooks/testes_preliminares_colab.ipynb` no Google Colab.
2. Execute as células na ordem. A instalação pode exigir reiniciar a sessão antes dos imports.
3. Quando solicitado, envie `HPLC_data.csv`, disponível no projeto do Trabalho 1 em `data/HPLC_data.csv`. Se o repositório estiver clonado no ambiente, o notebook também procura o CSV em caminhos relativos.
4. Execute os testes sequencialmente e salve uma cópia do notebook com as saídas reais.

Os testes usam CPU. No ambiente relatado, havia Python 3.13.18, duas CPUs e aproximadamente 12,67 GB de RAM. Tempos, pipelines e métricas podem variar. O limite de busca não garante o mesmo tempo de parede total, devido ao ajuste final e ao encerramento dos processos.

## Resultados preliminares relatados

| Ferramenta | Limite de busca | Tempo observado | Observação |
|---|---|---|---|
| TPOT 1.1.0 | 5 minutos | 5,18 minutos | F1-macro 0,7146; acurácia 0,9977 |
| AutoGluon 1.6.3 | 300 segundos | 1,80 minuto | Ensemble selecionado; uma das nove classes foi descartada automaticamente |

No AutoGluon, o ensemble apresentou `score_val=0,998868` e `score_test=0,887394` ao chamar `leaderboard(valid_df)`. Nesse comando, `score_test` refere-se à validação preliminar fornecida, e não ao teste reservado do Trabalho 1. Os valores não devem ser comparados diretamente com os do TPOT: houve diferença no tratamento das classes. Veja o registro em `resultados/`.

## Resultado oficial do TPOT

O protocolo oficial foi executado com TPOT 1.1.0, F1-macro como objetivo, validação cruzada estratificada de cinco folds no treino e avaliação externa na validação. O budget de 5 minutos foi repetido nas cinco sementes; os budgets de 30 minutos e 60 minutos foram executados com a seed 42. O melhor budget na validação foi o de 60 minutos.

| Budget | Repetições | F1-macro na validação |
|---|---:|---:|
| 5 min | 5 | 0,5956 (média; DP 0,3324) |
| 30 min | 1 | 0,8672 |
| 60 min | 1 | 0,8892 |

No teste reservado, consultado uma única vez após a seleção, o TPOT alcançou F1-macro `0,9683`, F1-weighted `0,9996` e acurácia `0,9996`. Na ablação de 30 minutos, o F1-macro foi `0,8892` tanto com quanto sem os atributos derivados. Os artefatos estão em `resultados/tpot/`.

## Ajustes necessários antes do experimento oficial

- Recuperar os mesmos índices de treino/validação/teste 60/20/20 do Trabalho 1, suas oito classes e colunas descartadas; conferir por asserções. O teste preliminar usou nove classes e divisão 80/20.
- Fornecer somente treino às ferramentas. No teste preliminar, a validação foi entregue ao AutoGluon como `tuning_data`, o que não atende ao protocolo oficial.
- Ajustar imputação e codificação dentro dos folds internos. A preparação preliminar do TPOT usou a base inteira antes da divisão e não é adequada para estimar desempenho sem vazamento.
- Auditar classes raras, seleção de modelos, pré-processamento e ensembles. O CatBoost foi ignorado por dependência ausente na execução relatada.
- Executar os budgets de 5 min, 30 min e 1 h, sementes e medições exigidos no enunciado; reservar teste para uma avaliação por sistema.
- Acrescentar os notebooks oficiais, arquivos de ambiente completos, logs, predições e auditorias. O repositório deve permitir reprodução no Colab e localmente.

O código nesta etapa documenta a exploração inicial; seus resultados não estabelecem superioridade entre ferramentas.

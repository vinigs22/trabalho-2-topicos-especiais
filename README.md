# Trabalho 2 — Aprendizado de Máquina Automatizado

Disciplina C318 — Tópicos Especiais II, Inatel. Equipe 4.

Integrantes: Gabriel Lopes Silva, Lucas Caetano Reis, Pedro Tiso Vinhas Mesquita e Vinicius Gonçalves Simoni.

## Etapa atual

Este repositório registra os testes preliminares realizados no Google Colab em 12/11/2026 para a ficha de acompanhamento. As ferramentas escolhidas são **TPOT 1.1.0** (busca sobre pipelines scikit-learn) e **AutoGluon Tabular 1.6.3** (família de gradient boosting automatizado na tabela do enunciado).

Os testes confirmaram instalação, importação e treinamento no CSV. **Ainda não constituem a comparação oficial com o Trabalho 1.** O notebook foi reconstruído a partir das células e saídas compartilhadas pela equipe; não é uma exportação do caderno original e não contém saídas de execução fabricadas.

## Arquivos

- `notebooks/testes_preliminares_colab.ipynb`: instalação, preparação e testes das duas ferramentas, com os ajustes de API usados no Colab.
- `data/README.md`: origem e localização do CSV.
- `resultados/testes_2026-11-12.md`: resultados relatados, falhas e limitações.
- `requirements-autogluon.txt` e `requirements-tpot.txt`: versões das ferramentas testadas; não representam um congelamento completo do ambiente.

## Como executar

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

## Ajustes necessários antes do experimento oficial

- Recuperar os mesmos índices de treino/validação/teste 60/20/20 do Trabalho 1, suas oito classes e colunas descartadas; conferir por asserções. O teste preliminar usou nove classes e divisão 80/20.
- Fornecer somente treino às ferramentas. No teste preliminar, a validação foi entregue ao AutoGluon como `tuning_data`, o que não atende ao protocolo oficial.
- Ajustar imputação e codificação dentro dos folds internos. A preparação preliminar do TPOT usou a base inteira antes da divisão e não é adequada para estimar desempenho sem vazamento.
- Auditar classes raras, seleção de modelos, pré-processamento e ensembles. O CatBoost foi ignorado por dependência ausente na execução relatada.
- Executar os budgets de 5 min, 30 min e 1 h, sementes e medições exigidos no enunciado; reservar teste para uma avaliação por sistema.
- Acrescentar os notebooks oficiais, arquivos de ambiente completos, logs, predições e auditorias. O repositório deve permitir reprodução no Colab e localmente.

O código nesta etapa documenta a exploração inicial; seus resultados não estabelecem superioridade entre ferramentas.

# Trabalho 2 — Aprendizado de Máquina Automatizado

Disciplina C318 — Tópicos Especiais II, Inatel. Equipe 4.

Integrantes: Gabriel Lopes Silva, Lucas Caetano Reis, Pedro Tiso Vinhas Mesquita e Vinicius Gonçalves Simoni.

## Etapa atual

O pipeline manual e o TPOT oficial já possuem resultados versionados. O protocolo oficial do **AutoGluon Tabular 1.6.3** está implementado e pronto para as execuções longas no Google Colab; seus resultados oficiais só existirão depois que o notebook 03 for executado com as chaves correspondentes.

O notebook preliminar de 12/11/2026 continua no repositório apenas como registro da ficha de acompanhamento. Seus números usam outro split e não constituem a comparação oficial com o Trabalho 1.

## Arquivos

- `notebooks/01_pipeline_manual.ipynb`: **protocolo oficial** — dados e filtros do Trabalho 1, partição 60/20/20 fixa, pipeline manual de referência (T1 corrigido), baseline de classe majoritária, métricas, tempo, memória e importância por permutação. Resultados em `resultados/manual/`.
- `src/c318_common.py`: funções compartilhadas por todos os notebooks oficiais (leitura do CSV, filtros do T1, partição, asserções, métricas e medição de tempo/memória).
- `notebooks/02_tpot_oficial.ipynb`: experimento oficial do **TPOT**. Executa a curva de budgets, seleciona o budget exclusivamente pela validação, realiza a ablação dos atributos derivados, calcula importância por permutação e reserva o teste para uma única consulta final.
- `src/tpot_protocol.py`: prepara a entrada do TPOT e concentra a configuração, a busca e a auditoria da ferramenta.
- `notebooks/03_autogluon_oficial.ipynb`: protocolo oficial do **AutoGluon** — sete runs, ablação pareada, auditoria do ensemble/modelo vencedor e bloqueio da segunda consulta ao teste.
- `src/autogluon_protocol.py`: preparação dos 17/15 atributos, holdout interno por seed, treino, medição, avaliação, persistência e seleção do budget do AutoGluon.
- `tests/test_autogluon_protocol.py`: testes do preparo, splits internos e regra de seleção, sem iniciar buscas AutoML.
- `data/HPLC_data.csv` e `data/splits/split_indices.csv`: CSV do Trabalho 1 e índices de treino/validação/teste usados por todos os sistemas.
- `requirements-manual.txt`: versões exatas do pipeline manual.
- `notebooks/testes_preliminares_colab.ipynb`: instalação, preparação e testes das duas ferramentas, com os ajustes de API usados no Colab.
- `data/README.md`: origem e localização do CSV.
- `resultados/testes_2026-11-12.md`: resultados relatados, falhas e limitações.
- `requirements-autogluon.txt` e `requirements-tpot.txt`: ferramentas e dependências diretas dos notebooks; cada execução oficial também salva o `pip freeze` completo em seus resultados.

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
3. Para as sete execuções da curva, altere somente `RUN_VALIDATION = True`. O notebook usa as cinco sementes no budget de 5 minutos e somente a seed 42 nos budgets de 30 e 60 minutos.
4. Após a escolha gravada em `resultados/tpot/budget_selecionado.json`, execute separadamente `RUN_DERIVED_ABLATION = True` e `RUN_IMPORTANCE = True`.
5. Somente quando toda a seleção estiver congelada, altere `RUN_FINAL_TEST = True`. A célula impede nova consulta se `resultados/tpot/metricas_teste.csv` já existir.

O TPOT recebe apenas transformações determinísticas por registro antes da busca. O notebook usa o espaço oficial `linear-light` do TPOT 1.1.0, pois o espaço personalizado com `ColumnOneHotEncoder` apresentou erro interno na seleção da população nessa versão. Essa escolha fica registrada para auditoria; a configuração efetiva deve ser conferida na saída da pipeline e no relatório.

### AutoGluon (notebook oficial)

1. No Google Colab em CPU, envie somente `notebooks/03_autogluon_oficial.ipynb`. A célula de bootstrap clona automaticamente a branch oficial rasa em `/content`, obtém os dados e módulos auxiliares e instala o AutoGluon 1.6.3 antes dos imports científicos.
2. Execute primeiro com todas as chaves `RUN_* = False` para conferir ambiente, dados, split e as oito classes. Não é necessário clonar o repositório, mudar o diretório de trabalho ou executar `pip install` manualmente.
3. Execute `RUN_SMOKE = True` isoladamente. O smoke usa somente parcelas do treino oficial e não produz resultado comparável.
4. Execute `RUN_VALIDATION = True`. São sete runs com atributos derivados: cinco seeds em 5 minutos e a seed 42 em 30 e 60 minutos. Runs concluídos são preservados e ignorados em retomadas.
5. Depois que `budget_selecionado.json` for criado, execute separadamente `RUN_DERIVED_ABLATION = True` e `RUN_AUDIT = True`. A ablação reaproveita o run de 30 minutos com atributos e treina apenas o par sem atributos.
6. Revise a auditoria e só então execute `RUN_FINAL_TEST = True` uma vez. O notebook recusa uma segunda consulta ao teste.

Para execução sem supervisão, deixe as cinco flags de etapa como `False`, defina somente `RUN_COMPLETE_PIPELINE = True` e use **Executar tudo**. O notebook respeita a mesma ordem segura, não repete smoke, runs, auditoria ou teste já concluídos e continua a matriz após uma falha isolada. Em uma nova execução, predictors parciais são movidos para `resultados/autogluon/modelos_parciais_preservados/`, sem apagar evidências, e apenas o run ausente é treinado novamente.

O holdout entregue como `tuning_data` é criado exclusivamente dentro das 7.806 linhas do treino oficial. A validação de 2.602 linhas nunca entra no `fit`; ela compara os runs e congela o budget. O limiar de classes do AutoGluon é reduzido para 1 porque a menor classe possui oito exemplos no treino. Os predictors, potencialmente grandes, ficam em `resultados/autogluon/modelos/` e são ignorados pelo Git; CSVs, JSONs, logs exportados e a análise permanecem versionáveis.

Para conferir a lógica sem instalar ou treinar o AutoGluon, desde que as dependências tabulares estejam disponíveis:

```bash
pytest -q tests/test_autogluon_protocol.py
```

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

## AutoGluon oficial: trabalho ainda pendente

- Executar no Colab o smoke, os sete runs oficiais e o run adicional da ablação sem atributos derivados.
- Revisar a seleção congelada e a auditoria gerada antes de liberar a consulta única ao teste.
- Versionar `resultados/autogluon/`, exceto o diretório pesado `modelos/`, e preencher esta seção com os resultados reais. Não devem ser inferidas métricas a partir do teste preliminar.

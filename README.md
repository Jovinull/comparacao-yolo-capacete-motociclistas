# Laboratório de comparação YOLO para detecção de capacete

Repositório de código para comparar YOLOv8n e YOLO11n no mesmo conjunto de vídeos, com a mesma configuração de treino e o mesmo equipamento. O repositório contém apenas scripts, configurações e instruções do experimento; manuscrito, artigos e documentos da disciplina ficam fora dele.

## Escopo do experimento

O conjunto de dados usa quatro classes, nesta ordem:

| ID | Classe | Anotação |
|---:|---|---|
| 0 | `motocicleta` | Caixa da motocicleta. |
| 1 | `motociclista_com_capacete` | Caixa de cada condutor ou garupa usando capacete. |
| 2 | `motociclista_sem_capacete` | Caixa de cada condutor ou garupa sem capacete. |
| 3 | `capacete` | Caixa de cada capacete visível, esteja sendo usado ou não. |

O estado do motociclista é anotado na caixa da pessoa. A classe `capacete` descreve o objeto e não substitui a anotação do estado de uso. Os casos incertos devem seguir o manual de anotação do grupo e não receber um rótulo forçado.

## Protocolo implementado

- O manifesto registra `video_id` e partição de cada quadro; o validador impede que um vídeo apareça em mais de uma partição.
- O validador verifica arquivos, rótulos YOLO, quatro classes e a presença de cada classe em treino e validação. Classes ausentes no teste geram aviso.
- Os dois modelos são treinados sequencialmente com os mesmos parâmetros.
- Depois do treino, uma grade fixa de limiares é avaliada na validação. Para cada modelo, o limiar que maximiza F1 macro nas quatro classes é congelado antes de abrir o teste; em empate, escolhe-se o maior limiar.
- Antes do treino, o validador lê os rótulos das três partições para checar integridade e distribuição das classes. Ele não calcula métricas do modelo no teste; predições e métricas de teste só são geradas depois de os dois limiares serem escolhidos na validação.
- O teste reporta mAP@0,5:0,95, mAP@0,5 e velocidade do avaliador, além de precisão, revocação, F1, matriz de confusão, contagens e taxas de falsos positivos e negativos por classe no limiar congelado.
- A avaliação usa lote 1. O relatório inclui tempo de treino, tamanho dos pesos, pico amostrado de memória de CPU, pico de memória alocada/reservada de GPU e FPS do detector e do avaliador. O FPS inclui pré e pós-processamento dos quadros, mas não a decodificação do arquivo de vídeo.
- A revocação de `motociclista_sem_capacete` é calculada por vídeo e comparada com bootstrap pareado entre vídeos.
- Se o manifesto tiver anotações de condições visuais, o relatório também separa métricas por condição.

O conjunto de dados próprio e os resultados experimentais ainda não estão neste repositório. Portanto, os pesos COCO carregados pelo smoke test verificam apenas a instalação e a inferência, não a detecção de capacete nem o desempenho do experimento.

## Requisitos e ambiente

Para a verificação local em CPU, crie um ambiente virtual e instale `requirements-local.txt`:

```powershell
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements-local.txt
.\env\Scripts\python.exe scripts/smoke_test.py
```

O treino exige GPU NVIDIA. Use um runtime com PyTorch e CUDA compatíveis já instalados e, depois, execute `python -m pip install -r requirements-cloud.txt`. Esse arquivo não instala uma versão CPU do PyTorch. Registre a versão de Python, PyTorch, CUDA, Ultralytics, GPU e os parâmetros usados em cada execução. `requirements-lock.txt` corresponde ao ambiente local de CPU e não deve ser usado como lock do ambiente CUDA.

## Dados

Copie `configs/data.example.yaml` para `data/helmet/data.yaml`, ajuste `path` e organize os arquivos:

```text
data/helmet/
  data.yaml
  manifest.csv
  images/
    train/  val/  test/
  labels/
    train/  val/  test/
```

Cada linha do manifesto contém `image_path,video_id,split`; a coluna opcional `conditions` aceita etiquetas separadas por ponto e vírgula: `low_light`, `occlusion`, `far_camera`, `low_resolution` e `passenger`. Exemplo:

```csv
image_path,video_id,split,conditions
images/train/video_01_000001.jpg,video_01,train,low_light;passenger
```

Os rótulos seguem o formato YOLO `class_id x_center y_center width height`, com coordenadas normalizadas entre 0 e 1. Use os IDs da tabela de classes acima e inclua um `.txt` vazio para quadros sem objetos. Todos os quadros do mesmo vídeo devem ficar na mesma partição.

Vídeos, imagens, rótulos e `data/helmet/data.yaml` são ignorados pelo Git. Só use dados com origem e autorização confirmadas pelo grupo.

## Executar

Depois que o grupo confirmar os dados, as anotações, a GPU e a configuração:

```powershell
python scripts/validate_dataset.py --data data/helmet/data.yaml
python scripts/run_comparison.py --config configs/experiment.yaml
```

`configs/experiment.yaml` contém valores iniciais. Revise-os antes do treino e preserve a configuração aprovada junto aos resultados. Para usar CPU apenas em uma checagem de fluxo, passe `--device cpu`; isso não substitui o treino comparável em GPU.

Relatórios, gráficos e pesos são gravados em `runs/` e `weights/`, que não entram no Git. Cada execução guarda a configuração, a validação do conjunto, as versões e o hardware, os limiares escolhidos na validação e os resultados gerais, por classe, por vídeo e por condição disponível.

## Pendências do grupo

Confirmar fonte, autorização e contexto dos vídeos; aprovar o manual de anotação; revisar a amostra de rótulos; definir a GPU e os parâmetros comuns; e revisar os resultados do piloto antes do experimento principal. Este repositório ainda não contém vídeos autorizados nem resultados do grupo.

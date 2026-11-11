# Laboratório de comparação YOLO para uso de capacete

Repositório do experimento que compara YOLOv8n e YOLO11n sob parâmetros e equipamento controlados. Inclui o protocolo principal de quatro classes, um preparo reprodutível do conjunto HELMET publicado no OSF e o registro do piloto EdgeVision. Dados brutos, imagens preparadas, pesos e execuções ficam fora do Git.

## Trilhas experimentais

### Protocolo principal: quatro classes

O desenho principal, definido no planejamento do grupo, prevê quatro classes anotadas em vídeos de um contexto brasileiro:

| ID | Classe | O que recebe caixa |
|---:|---|---|
| 0 | `motocicleta` | Motocicleta. |
| 1 | `motociclista_com_capacete` | Cada condutor ou garupa usando capacete. |
| 2 | `motociclista_sem_capacete` | Cada condutor ou garupa sem capacete. |
| 3 | `capacete` | Cada capacete visível, usado ou não. |

O estado de uso é anotado na caixa da pessoa. Esse protocolo continua em `configs/experiment.yaml`; os vídeos locais, a autorização de uso, o manual de anotação e a GPU compartilhada ainda precisam ser confirmados pelo grupo.

### Comparação complementar: HELMET do OSF

O conjunto [HELMET](https://osf.io/4pwj8/) tem 91.000 quadros em 910 clipes de Myanmar e 283.377 caixas de motocicletas. As anotações são por motocicleta e codificam o estado dos ocupantes em etiquetas compostas; não há caixas separadas de pessoa ou capacete. O perfil converte essas anotações em duas classes de estado agregado por motocicleta e mantém a divisão oficial por vídeo. Ele é uma comparação complementar, não substitui o protocolo brasileiro de quatro classes. Leia [HELMET_OSF.md](HELMET_OSF.md) antes de preparar ou interpretar essa base.

### Piloto auxiliar: EdgeVision

O piloto EdgeVision comparou YOLOv8n e YOLO11n em 2.392 imagens e três classes, em CPU. O relatório e os arquivos de resultados estão versionados em `results/edgevision/20261030_225248/`. Esse piloto verificou o fluxo de treinamento e avaliação, mas não substitui o protocolo principal nem a comparação no conjunto HELMET. Veja [EDGEVISION_CPU.md](EDGEVISION_CPU.md).

## Protocolo de comparação

- `scripts/validate_dataset.py` valida os arquivos, nomes e rótulos YOLO, exige imagens e rótulos nas três partições e bloqueia a presença do mesmo `video_id` em mais de uma partição.
- `scripts/run_comparison.py` treina os dois modelos sequencialmente com a configuração comum.
- Após o treino, seleciona-se na validação o limiar de confiança que maximiza F1 macro na grade definida; os limiares são congelados antes de consultar o teste.
- O relatório inclui mAP@0,5 e mAP@0,5:0,95, precisão, revocação, F1, matriz de confusão, contagens de erros, custos de treino e inferência, uso de memória e análise pareada por vídeo para a classe-alvo configurada.
- O validador lê os rótulos do teste para conferir integridade e distribuição, mas não calcula métricas do modelo no teste. As previsões e métricas do teste só são produzidas depois de os dois limiares serem escolhidos na validação.

## Ambiente

Para instalar o ambiente local de CPU e preparar dados:

```powershell
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements-local.txt
```

O treino comparativo requer GPU NVIDIA. Use um runtime com PyTorch e CUDA compatíveis já instalados e depois instale `requirements-cloud.txt`; esse arquivo instala as dependências do projeto sem substituir o PyTorch/CUDA do runtime. `requirements-lock.txt` corresponde ao ambiente local de CPU, não ao ambiente CUDA.

Registre em cada execução as versões de Python, PyTorch, CUDA, Ultralytics, GPU, driver e os parâmetros efetivamente usados.

## Preparar os dados

Para o protocolo principal, organize os dados assim:

```text
data/helmet/
  data.yaml
  manifest.csv
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Cada linha de `manifest.csv` contém `image_path,video_id,split`; a coluna opcional `conditions` pode conter etiquetas separadas por ponto e vírgula: `low_light`, `occlusion`, `far_camera`, `low_resolution` e `passenger`. Todos os quadros do mesmo vídeo devem ficar na mesma partição. Cada imagem precisa de um `.txt` de rótulo, vazio quando não houver objetos.

Para baixar e preparar o conjunto HELMET do OSF, use as instruções em [HELMET_OSF.md](HELMET_OSF.md). Para repetir o piloto auxiliar EdgeVision, consulte [EDGEVISION_CPU.md](EDGEVISION_CPU.md).

## Executar o protocolo principal

Depois que os vídeos e as anotações do grupo estiverem prontos e a GPU for definida:

```powershell
python scripts/validate_dataset.py --data data/helmet/data.yaml
python scripts/run_comparison.py --config configs/experiment.yaml
```

Revise os valores iniciais de `configs/experiment.yaml` com o grupo antes de uma execução longa. A opção `--device cpu` serve para checar o fluxo em CPU e não substitui a comparação final em GPU.

## Arquivos registrados e dados ignorados

O Git registra scripts, configurações, instruções e o relatório do piloto. `.gitignore` exclui imagens, rótulos, arquivos baixados, pesos e execuções em `data/`, `weights/` e `runs/`. Não envie os arquivos brutos do OSF nem imagens locais para este repositório; cite o conjunto original e preserve os manifestos e hashes dos arquivos usados.

## Pendências do protocolo principal

Confirmar com o grupo a fonte e autorização dos vídeos brasileiros; aprovar as regras e revisar uma amostra de rótulos; definir a GPU e os parâmetros comuns; e executar o piloto nos dados do contexto principal. Os resultados do EdgeVision são apenas auxiliares, e a base HELMET do OSF representa outro contexto e outra definição de classes.

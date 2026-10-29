# Laboratório de comparação YOLO para detecção de capacete

Repositório de código para comparar YOLOv8n e YOLO11n no mesmo conjunto de vídeos, com a mesma configuração de treino e o mesmo equipamento. Este repositório contém scripts e configurações do experimento; manuscrito, artigos e documentos da disciplina ficam fora dele.

## Protocolo implementado

- O manifesto registra `video_id` e partição de cada quadro. `validate_dataset.py` impede que quadros do mesmo vídeo apareçam em treino, validação e teste.
- `run_comparison.py` treina os dois modelos sequencialmente com parâmetros comuns e só avalia o conjunto de teste depois que ambos os treinos terminam.
- Métrica primária: mAP@0,5:0,95. O relatório também registra mAP@0,5, precisão, recall, métricas por classe e velocidade medida na validação.
- A análise secundária calcula recall de `sem_capacete` por vídeo e um intervalo bootstrap pareado entre os vídeos do teste. O limiar de confiança e o IoU dessa análise estão explícitos na configuração.

**Estado atual:** o código está preparado, mas o conjunto de dados ainda não está neste repositório. Os valores em `configs/experiment.yaml` são uma configuração inicial; o grupo deve aprová-los antes do treino. Não há resultados experimentais registrados.

## Dados esperados

Copie `configs/data.example.yaml` para `data/helmet/data.yaml`, ajuste `path` se necessário e organize os quadros e rótulos assim:

```text
data/helmet/
  data.yaml
  manifest.csv
  images/
    train/  val/  test/
  labels/
    train/  val/  test/
```

Cada linha do `manifest.csv` contém `image_path,video_id,split`. `image_path` é relativo à raiz do conjunto, por exemplo `images/train/video_01_000001.jpg`. Os rótulos seguem o formato YOLO: `class_id x_center y_center width height`, coordenadas normalizadas entre 0 e 1. IDs de classe: `0=com_capacete`, `1=sem_capacete`. Inclua um `.txt` vazio para quadros sem objetos.

Vídeos, imagens e rótulos são dados do grupo e ficam ignorados pelo Git. O manifesto de exemplo em `configs/manifest.example.csv` contém apenas o cabeçalho.

## Preparar e executar

No Windows, use o ambiente isolado já preparado ou crie um novo:

```powershell
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements-local.txt
.\env\Scripts\python.exe scripts/smoke_test.py
```

O smoke test só verifica carregamento e inferência em CPU; não treina nem mede qualidade. Para o experimento, instale `requirements-cloud.txt` em um runtime com GPU NVIDIA. Use a mesma GPU e versão de software para os dois modelos.

Depois que os dados e a configuração forem aprovados:

```bash
python scripts/validate_dataset.py --data data/helmet/data.yaml
python scripts/run_comparison.py --config configs/experiment.yaml
```

Os relatórios, gráficos e pesos treinados são salvos em `runs/` e não entram no Git. O `summary.json` registra configuração, versões, hardware, métricas gerais, métricas por classe, recall por vídeo e o intervalo bootstrap pareado.

## Decisões do grupo ainda necessárias

Confirmar vídeos e autorização de uso, regras de anotação para condutor/garupa/oclusões, divisão por vídeo, GPU e parâmetros comuns de treino. Ajuste `configs/experiment.yaml` depois dessas decisões e preserve a configuração usada junto dos resultados.

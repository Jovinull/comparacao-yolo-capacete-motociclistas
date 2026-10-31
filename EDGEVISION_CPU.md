# Perfil EdgeVision para CPU (piloto auxiliar)

Este perfil prepara uma comparação de YOLOv8n e YOLO11n nas imagens públicas EdgeVision, com teto de 12 horas na máquina local. Ele não substitui o protocolo principal de vídeos e quatro classes.

## Dataset

O perfil baixa a versão 1 do EdgeVision Dataset, com 2.392 imagens e as classes `BikeWithRider`, `NoHelmet` e `Helmet`. A licença publicada é CC BY 4.0. O dataset não informa `video_id`; portanto, a divisão 80/10/10 é por imagem e não permite análise pareada por vídeo.

Citação: Gajjar, K.; Patel, N.; Patel, R.; Patel, K.; Dabhi, C.; Trivedi, H.; Goyani, M.; Bhatt, C. *EdgeVision Dataset*. Mendeley Data, v1, 2026. DOI: [10.17632/j82bnw7gsr.1](https://doi.org/10.17632/j82bnw7gsr.1). Atribuição CC BY 4.0.

Os arquivos originais baixados ficam em `data/edgevision/source/`, preservados com os hashes oficiais. O preparo cria cópias limpas em `data/edgevision/prepared/images/` e rótulos em `data/edgevision/prepared/labels/`, corrigindo JPEGs sem marcador final e aplicando rotação EXIF sem sobrescrever os originais. O registro dessas cópias fica em `preprocessing_report.json`. Na conversão, caixas que ultrapassam a borda são ajustadas ao limite da imagem, e caixas degeneradas são ignoradas; cada caso aparece em `annotation_conversion_report.json`.

## Preparar e executar

No PowerShell, a partir da pasta do projeto:

```powershell
.\env\Scripts\python.exe scripts\download_edgevision.py
.\env\Scripts\python.exe scripts\run_edgevision_comparison.py --config configs\edgevision_12h.yaml
```

Antes de uma execução longa, confira o dataset sem iniciar o treino:

```powershell
.\env\Scripts\python.exe scripts\run_edgevision_comparison.py --config configs\edgevision_12h.yaml --check-data-only
```

O downloader verifica tamanho e SHA-256 dos arquivos, converte as caixas COCO em rótulos YOLO e gera uma divisão estratificada e reproduzível com semente 42. Se o download for interrompido, execute o primeiro comando novamente; os arquivos já íntegros serão reaproveitados.

## Orçamento local

O perfil usa CPU, lote 4, duas workers, entrada 640 e até 100 épocas. O runner limita cada treino a 4,5 horas, usando um callback compatível com o Ultralytics 8.3.75 instalado no ambiente. A reserva de três horas cobre validação, teste e encerramento. O limite é verificado no fim de cada época, então o treino pode ultrapassar o teto por até uma época.

O teto distribui tempo igual por modelo; os modelos podem completar números diferentes de épocas. O relatório registra o tempo real e as épocas concluídas. Esse limite de 12 horas é um alvo operacional, não uma garantia rígida caso o sistema fique ocupado ou a avaliação demore além da reserva.

Cada execução cria uma pasta própria em `runs/edgevision/` e preserva o YAML usado, resumo JSON/CSV, métricas por classe, versões do ambiente, lista com hashes dos arquivos, partições exatas e relatório de conversão. Os subdiretórios de treino guardam `results.csv`, curvas e pesos `best.pt`/`last.pt`; as avaliações guardam matrizes/curvas e predições JSON para validação e teste. O comando não envia os dados nem os pesos ao GitHub.

## Limite metodológico

Esta base contém fotografias com três classes, enquanto o laboratório principal usa quatro classes e vídeos com identificação de origem. Os resultados deste perfil servem para verificar o fluxo e comparar os modelos nesta base de imagens; não devem ser apresentados como resultados do experimento de videomonitoramento planejado no artigo.

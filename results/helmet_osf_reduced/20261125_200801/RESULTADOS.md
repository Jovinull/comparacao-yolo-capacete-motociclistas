# HELMET OSF - protocolo reduzido (execução 20261125_200801)

**Não comparável ao protocolo completo** (`configs/helmet_osf.yaml`). Este perfil
(`configs/helmet_osf_reduced.yaml`) usa entrada 640, lote 8, 10 épocas e paciência 5,
em uma RTX 3050 Laptop (4 GB) com 8 GB de RAM.

Teste reservado; limiar de confiança 0,45 escolhido na validação.

| Métrica | YOLOv8n | YOLO11n |
|---|---|---|
| mAP50 | 0,871 | 0,859 |
| mAP50-95 | 0,752 | 0,741 |
| Precisão (macro) | 0,795 | 0,785 |
| Revocação (macro) | 0,801 | 0,788 |
| F1 (macro) | 0,797 | 0,786 |
| Revocação, ocupante sem capacete | 0,745 | 0,735 |
| Latência de inferência (ms) | 24,1 | 23,0 |
| FPS do pipeline | 32,6 | 35,3 |

Bootstrap pareado por vídeo (155 vídeos), revocação da classe com ocupante sem capacete,
YOLO11n menos YOLOv8n: -0,016 (IC 95% [-0,035; 0,001]). O intervalo inclui zero.

## Ressalvas

- Com 10 épocas, os modelos podem não ter convergido.
- Dados de Mianmar, com a motocicleta como unidade rotulada; não representam o contexto brasileiro.
- Latência medida com pouca memória livre; use apenas como comparação relativa.
- A avaliação foi refeita com `--eval-only` após um travamento por falta de memória; o treino
  não foi repetido. Pico de memória do treino não foi medido, e o tempo de treino vem do
  `results.csv` (o do YOLOv8n inclui um trecho lento com o notebook na bateria).
- Caminhos absolutos foram convertidos em relativos nos JSON arquivados.

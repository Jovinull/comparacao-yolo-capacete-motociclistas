# Comparação adicional no conjunto HELMET (OSF)

Este perfil prepara o conjunto público [HELMET dataset](https://osf.io/4pwj8/) e compara YOLOv8n com YOLO11n usando a divisão oficial por vídeo. O preparo e os resultados ficam fora do Git; o repositório guarda o código, a configuração e as instruções.

## O que há no conjunto

O OSF publica 910 clipes de videomonitoramento de Myanmar, com 100 quadros por clipe, 10 fps e resolução de 1920 × 1080: 91.000 quadros no total. O pacote inclui 283.377 anotações de caixas de motocicletas, cada uma associada ao estado de uso de capacete dos ocupantes anotados. A divisão fornecida pelo conjunto contém 636 vídeos de treino, 92 de validação e 182 de teste. O script preserva esses grupos de vídeo para evitar que quadros do mesmo clipe apareçam em partições diferentes.

Após agrupar as etiquetas compostas nas duas classes do perfil, a distribuição das caixas fica assim:

| Partição | Com todos usando capacete | Com pelo menos um sem capacete | Total de caixas |
|---|---:|---:|---:|
| Treino | 121.107 | 78.496 | 199.603 |
| Validação | 17.312 | 11.933 | 29.245 |
| Teste | 33.714 | 20.815 | 54.529 |
| **Total** | **172.133** | **111.244** | **283.377** |

Fonte: Lin, H.; Deng, J. D.; Albers, D.; Siebert, F. W. *Helmet Use Detection of Tracked Motorcycles Using CNN-Based Multi-Task Learning*. IEEE Access, 8, 162073–162084, 2020. [DOI 10.1109/ACCESS.2020.3021357](https://doi.org/10.1109/ACCESS.2020.3021357). Conjunto publicado no [OSF](https://osf.io/4pwj8/), sob licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Como as classes são construídas

O arquivo de anotação fornece uma caixa por motocicleta e uma etiqueta composta, como `DHelmetP1NoHelmet`. Não existem caixas individuais de motociclistas ou capacetes. Por isso, este perfil converte cada caixa de motocicleta em uma destas duas classes:

| ID | Classe gerada | Regra aplicada à etiqueta original |
|---:|---|---|
| 0 | `motocicleta_todos_com_capacete` | Todos os ocupantes anotados na motocicleta estão marcados com `Helmet`. |
| 1 | `motocicleta_com_ocupante_sem_capacete` | Pelo menos um ocupante anotado está marcado com `NoHelmet`. |

Esta tarefa mede a localização de motocicletas e a classificação do estado de capacete agregado por motocicleta. Ela não mede caixas de pessoas nem de capacetes e não substitui o protocolo principal de quatro classes definido em `configs/experiment.yaml`. Também não representa o contexto brasileiro previsto no artigo. Portanto, trate-a como uma comparação complementar em uma base pública, sem misturar seus resultados aos resultados da coleta local.

## Download e preparo

O pacote de imagens tem 28,86 GB compactados. Reserve cerca de 40 GB livres durante o preparo; o script processa uma parte de cada vez, verifica o SHA-256 fornecido pelo OSF e, por padrão, apaga cada ZIP depois de convertê-lo. Ao final, as imagens preparadas ocupam aproximadamente 30 GB. `--keep-archives` conserva os ZIPs originais e exige espaço adicional.

No PowerShell, a partir da raiz do repositório:

```powershell
python scripts/download_helmet_osf.py
python scripts/validate_dataset.py --data data/helmet_osf/data.yaml
```

O download pode ser retomado executando o mesmo comando. Se os sete ZIPs já estiverem em `data/helmet_osf/source/downloads/`, prepare os dados sem consultar o OSF:

```powershell
python scripts/download_helmet_osf.py --prepare-only
```

Para manter os ZIPs originais após o preparo:

```powershell
python scripts/download_helmet_osf.py --keep-archives
```

O script grava `data.yaml`, `manifest.csv`, as imagens e rótulos YOLO, além dos relatórios de origem e conversão em `data/helmet_osf/`. Essa pasta está ignorada pelo Git.

## Treino e avaliação

O perfil comum está em `configs/helmet_osf.yaml`: YOLOv8n e YOLO11n são treinados sequencialmente, com pesos pré-treinados, seed 42, entrada 1280, lote 4 e limite de 100 épocas com parada antecipada. A grade de confiança é selecionada na validação antes de avaliar o teste. O relatório inclui mAP, métricas por classe, custos computacionais e bootstrap pareado por vídeo para a revocação da classe com ocupante sem capacete.

O treino completo requer uma GPU NVIDIA e uma instalação de PyTorch com CUDA compatível com o equipamento. No ambiente CUDA:

```powershell
python -m pip install -r requirements-cloud.txt
python scripts/run_comparison.py --config configs/helmet_osf.yaml
```

Confirme as versões de Python, PyTorch, CUDA e Ultralytics antes de iniciar uma execução longa. Os relatórios e pesos ficam em `runs/helmet_osf/` e `weights/`, ambos fora do Git.

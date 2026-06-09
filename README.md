# Offline × Online: Avaliação do Aumento de Dados em MRI Volumétrica do Joelho *>>>(em andamento)<<<*

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Avaliação do impacto da **quantidade** de aumento de dados offline na classificação binária de volumes de ressonância magnética do joelho (3D DESS), com comparação direta ao aumento online estocástico.


Continuação direta de:

| [SBCAS-2025](https://github.com/thallescotta/treinamento-cnn-3d-2026) | ➡️ | [SBCAS-2026](https://github.com/thallescotta/sbcas2026) | ➡️ | **Este trabalho** |
|:---:|:---:|:---:|:---:|:---:|
| R3D-18, oversampling, k-fold | | Aumento online, EX1-EX10 | | Offline × Online, curva AUC × proporção |
| AUC ≈ 0,92 | | AUC ≈ 0,90 | | *(em andamento)* |


---

## Motivação

No SBCAS-2026, o aumento online estocástico não produziu diferenças estatisticamente significativas entre os experimentos EX1–EX10. Uma hipótese central é que a natureza estocástica do aumento online impede o controle preciso sobre *quanto* de variação o modelo efetivamente recebe.

Este projeto supera essa limitação ao adotar **aumento offline**: os volumes aumentados são pré-gerados e salvos em disco em proporções controladas (10%, 25%, 50%, 100%, 150%, 200%), permitindo construir a curva **AUC × proporção de aumento** e identificar o ponto de saturação.

---

**Literatura de embasamento:**
- [Shorten & Khoshgoftaar (2019)](https://github.com/thallescotta/offline-x-online-augmentation-knee-mri/blob/main/artigos/doi.org_10.1186_s40537-019-0197-0.pdf): revisam técnicas de aumento de dados para deep learning em imagens, incluindo transformações geométricas, manipulações de cor, GANs, aumento em espaço de características e decisões de projeto como tamanho final do dataset e test-time augmentation.
- [Perez & Wang (2017)](https://github.com/thallescotta/offline-x-online-augmentation-knee-mri/blob/main/artigos/arXiv1712.04621v1.pdf): comparam diferentes estratégias de aumento de dados para classificação de imagens, incluindo transformações tradicionais, GANs/style transfer e aumento aprendido por redes neurais.
- [Chlap et al. (2021)](https://github.com/thallescotta/offline-x-online-augmentation-knee-mri/blob/main/artigos/doi10_1111_1754-9485.13261.pdf): revisam técnicas de aumento de dados em imagens médicas de CT e MRI para aplicações de deep learning, abrangendo métodos básicos, deformáveis, baseados em deep learning e outras abordagens.




---

## Estrutura do projeto (em andamento)

```
augmentation-offline-2026/
│
├── generate_offline.py     # Gerador: lê .npy originais → gera volumes aumentados por ratio
│
├── EX_BASELINE.py          # Sem aumento (análogo ao EX4 do SBCAS-2026)
├── EX_OFFLINE_010.py       # Offline +10%
├── EX_OFFLINE_025.py       # Offline +25%
├── EX_OFFLINE_050.py       # Offline +50%
├── EX_OFFLINE_100.py       # Offline +100%
├── EX_OFFLINE_150.py       # Offline +150%
├── EX_OFFLINE_200.py       # Offline +200%
├── EX_ONLINE_COMPARE.py    # Online L2 (réplica do EX8 SBCAS-2026 : comparação direta)
│
├── analyze_results.py      # Curva AUC × proporção + testes t pareados + tabela
│
├── oai3d_offline/          # Módulo central
│   ├── __init__.py
│   ├── config.py           # ExperimentConfig (compatível com SBCAS-2026)
│   ├── dataset.py          # OriginalDataset, AugmentedDataset, CombinedDataset, OnlineAugDataset
│   └── runner.py           # Orquestrador: 5-fold, treino, avaliação, persistência
│
├── results/                # Gerado automaticamente (não versionado)
│   └── <exp_name>/
│       ├── fold_1/ … fold_5/
│       │   ├── y_true.npy
│       │   ├── y_prob.npy
│       │   ├── best_inner.pth
│       │   └── final_model.pth
│       └── summary.json
│
└── requirements.txt
```

---

## Dataset

Mesmo dataset dos trabalhos anteriores:

```
~/dataset/OAI-MRI-3DDESS/
  normal-3DESS-128-64.npy     # 1659 volumes (classe negative)
  abnormal-3DESS-128-64.npy   # 1317 volumes (classe positive)
```

Fonte: [OAI-MRI-3DDESS no Kaggle](https://www.kaggle.com/datasets/mohamedberrimi/oaimri3ddess) (Berrimi, 2022).  
Os dados **não** estão incluídos por restrições de licença.

---

## Ambiente (LASAI / deep-02)

- Hardware: 2× NVIDIA GeForce RTX 2080 Ti (12 GB cada)
- Python: 3.10.19
- PyTorch: 2.0.1 + CUDA 11.8 + cuDNN 8.9.5

---

## Execução passo a passo

### 1. Instalar dependências (Se necessário)

```bash
pip install -r requirements.txt
```

### 2. Gerar os volumes aumentados (uma única vez)

```bash
python generate_offline.py \
    --data_dir ~/dataset/OAI-MRI-3DDESS \
    --out_dir  ~/dataset/OAI-MRI-3DDESS-offline \
    --ratios 0.10 0.25 0.50 1.00 1.50 2.00 \
    --seed 42
```

Saída esperada em `~/dataset/OAI-MRI-3DDESS-offline/`:
```
ratio_010pct/normal_aug.npy      (165 volumes)
ratio_010pct/abnormal_aug.npy    (131 volumes)
ratio_025pct/...
...
manifest.json
```

### 3. Rodar os experimentos

```bash
# Baseline (sem aumento)
CUDA_VISIBLE_DEVICES=0,1 python EX_BASELINE.py

# Offline por proporção
CUDA_VISIBLE_DEVICES=0,1 python EX_OFFLINE_010.py
CUDA_VISIBLE_DEVICES=0,1 python EX_OFFLINE_025.py
CUDA_VISIBLE_DEVICES=0,1 python EX_OFFLINE_050.py
CUDA_VISIBLE_DEVICES=0,1 python EX_OFFLINE_100.py
CUDA_VISIBLE_DEVICES=0,1 python EX_OFFLINE_150.py
CUDA_VISIBLE_DEVICES=0,1 python EX_OFFLINE_200.py

# Online (comparação direta com SBCAS-2026 EX8)
CUDA_VISIBLE_DEVICES=0,1 python EX_ONLINE_COMPARE.py
```

### 4. Analisar resultados

```bash
python analyze_results.py
```

Gera:
- Terminal: tabela AUC por experimento + testes t pareados
- `results/comparison_table.csv`
- `results/figures/auc_vs_ratio.pdf` : figura principal para o artigo
- `results/figures/auc_vs_ratio.png`

---

## Protocolo experimental
 
Idêntico ao SBCAS-2026 para garantir comparabilidade direta:
 
| Parâmetro | Valor | Justificativa |
|---|---|---|
| Arquitetura | R3D-18 pré-treinada (Kinetics-400) | Melhor resultado no SBCAS-2026: AUC 88,49% (EX1) vs. 76,48% treinado do zero (EX2), p < 0,0001 |
| Otimizador | Adam, lr=1×10⁻⁴ | Padrão amplamente adotado para fine-tuning de CNNs médicas (Kingma & Ba, 2014) |
| Perda | BCEWithLogitsLoss + pos_weight dinâmico | Trata desbalanceamento (1,26:1) sem oversampling, calculado por fold sobre o inner_train |
| AMP | Habilitado (autocast + GradScaler) | Viabiliza batch=32 nas 2× RTX 2080 Ti (12 GB); sem AMP, OOM força batch=2 (EX3 SBCAS-2026) |
| Validação | 5-fold estratificado, seed=42 | Equilíbrio entre robustez estatística e custo computacional; estratificação preserva proporção de classes por fold |
| inner_val | 15% do outer_train | Reservado exclusivamente para early stopping, sem participar da atualização de pesos nem da avaliação final |
| Patience | 15 épocas | Permite convergência sem interrupção prematura; valor fixado no SBCAS-2026 após inspeção das curvas de loss |
| Max épocas | 60 | Limite superior conservador; na prática o early stopping encerra antes em todos os experimentos do SBCAS-2026 |
| Batch size | 32 | Máximo viável com AMP nas 2× RTX 2080 Ti; batches maiores degradam estimativa do gradiente (ver EX3) |
| Oversampling | Desativado | Substituído por pos_weight, evitando redundância física de dados e variáveis de confusão entre experimentos |
 
### Transformações de aumento (parâmetros L2 / EX8 SBCAS-2026)
 
| Transformação | Parâmetro | Justificativa |
|---|---|---|
| Flip axial | p=0,5 | Presente em todos os experimentos do SBCAS-2026; simula inversão lateral do posicionamento do joelho |
| Rotação axial | ±10° | Nível L2 do SBCAS-2026, melhor AUC média (90,08%); dentro do limite anatômico de ±20° (Chlap et al., 2021) |
| Translação axial | ±10% | Nível L2 do SBCAS-2026; simula variação de posicionamento durante a aquisição (Shorten & Khoshgoftaar, 2019) |
| Ruído gaussiano | σ=0,02 | Nível L2 do SBCAS-2026; modela variação de equipamento e protocolo de MRI (Chlap et al., 2021) |
| p_apply (geom + ruído) | 0,5 | Probabilidade de aplicação por amostra; mesma do SBCAS-2026, garante que nem toda amostra é transformada |
 
Os parâmetros L2 (EX8) foram escolhidos por terem produzido a maior AUC média no SBCAS-2026 (90,08 ± 1,26%), tornando-os o ponto de referência natural para a comparação offline × online.

---

## Artefatos por experimento

Cada `EX_*.py` gera em `results/<exp_name>/`:

```
fold_1/
  y_true.npy       → rótulos verdadeiros (outer_test)
  y_prob.npy       → probabilidades preditas (para ROC/AUC)
  best_inner.pth   → checkpoint selecionado pelo early stopping
  final_model.pth  → modelo re-treinado no outer_train completo
summary.json       → AUC média, std, IC95% e resultados por fold
```

---

## Análise estatística

- Teste t pareado bicaudal (n=5 folds) entre cada condição e o baseline.
- IC95% via distribuição t de Student com 4 graus de liberdade.
- p-valores não corrigidos para múltiplas comparações (exploratório).
- Idêntico ao protocolo da Tabela 2 do SBCAS-2026 (Montgomery, 2017).

---


## Referências

- Berrimi, M. (2022). *OAI MRI 3D DESS dataset*. Kaggle.
- Chlap, P., Min, H., Vandenberg, N., Dowling, J., Holloway, L., & Haworth, A. (2021). A review of medical image data augmentation techniques for deep learning applications. *Journal of Medical Imaging and Radiation Oncology*.
- Hara, K., Kataoka, H., & Satoh, Y. (2018). Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P., Suleyman, M., & Zisserman, A. (2017). The Kinetics Human Action Video Dataset. *arXiv:1705.06950*.
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.
- Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley.
- Peterfy, C. G., Schneider, E., & Nevitt, M. (2008). The Osteoarthritis Initiative: report on the design rationale for the magnetic resonance imaging protocol for the knee. *Osteoarthritis and Cartilage*.
- Perez, L., & Wang, J. (2017). The Effectiveness of Data Augmentation in Image Classification using Deep Learning. *arXiv:1712.04621*.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*.



---

## Autor

**Thalles Cotta Fontainha** : `thalles.fontainha@aluno.cefet-rj.br`  
Programa de Pós-Graduação em Instrumentação e Óptica Aplicada - CEFET/RJ

**Orientador:** Prof. Felipe da R. Henriques - CEFET/RJ

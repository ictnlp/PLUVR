# Neural Machine Translation with Phrase-Level Universal Visual Representations

This is a PyTorch implementation for the ACL 2022 main conference paper [Neural Machine Translation with Phrase-Level Universal Visual Representations](https://arxiv.org/abs/2203.10299).

## Preparation

1. Clone this repository and install the dependencies:

```shell
git clone git@github.com:ictnlp/PLUVR.git
cd PLUVR/fairseq
pip install --editable ./
```

2. Data preparation:

Considering the complexity of data processing (noun phrase extraction using [spaCy](https://spacy.io/), visual grounding using [Yang et al., 2019](https://github.com/zyang-ur/onestage_grounding), feature extraction using [Detectron2](https://github.com/facebookresearch/detectron2), and visual retrieval), we have only released the necessary processed files for the model training and omitted some intermediate files. For example, the processed fairseq-format text data and other necessary files for the latent-variable model are in the `data-bin/multi30k_en_{de,fr}` folder.

Besides, for training of the latent-variable model, the visual features of all grounded regions are needed. 

~We will release it soon, and you can extract them using scripts in the `visual_grounding/` folder now.~

[upd:4/10] The visual features of all grounded regions `region_embedding.npy` can be downloaded via Baidu netdisk:

**Link:** https://pan.baidu.com/s/1IzGf-H8PnjYNOtZ4mU9F2w
**Password:** 8npa


## Training of latent-variable model

Train the latent-variable model:

```shell
cd vae/
python train.py --config configs/exp_512_64_multi30k_en_de.yaml
```

Save the phrase-guided visual representations of all grounded regions:

```shell
python get_latent.py --config configs/exp_512_64_multi30k_en_de.yaml
```

## Training of translation model

Train the model using 1 GPU:
```shell
sh train_multi30k_en_de.sh
```

## Inference

Average the last 5 checkpoints and generate the results, `test/test1/test2` indicate `Test2016/Test2017/MSCOCO`, respectively:
```shell
sh test.sh multi30k_en_de test multi30k_en_de
```

For evaluation, please refer to [sacreBLEU](https://github.com/mjpost/sacrebleu).

## Citation

If this repository is useful for you, please cite as:
```
@inproceedings{fang-and-feng-2022-PLUVR,
	title = {Neural Machine Translation with Phrase-Level Universal Visual Representations},
	author = {Fang, Qingkai and Feng, Yang},
	booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
	year = {2022},
}
```

## Contact

If you have any questions, feel free to contact me at `fangqingkai21b@ict.ac.cn`.

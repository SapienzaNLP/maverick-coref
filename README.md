<h1 align="center">
  Maverick Coref
</h1>
<div align="center">


[![Conference](https://img.shields.io/badge/ACL%202024%20Paper-red)](https://aclanthology.org/2024.acl-long.722.pdf)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Pip Package](https://img.shields.io/badge/🐍%20Python%20package-blue)](https://pypi.org/project/maverick-coref/)
[![git](https://img.shields.io/badge/Git%20Repo%20-yellow.svg)](https://github.com/SapienzaNLP/maverick-coref)
</div>


This is the official repository for [*Maverick:
Efficient and Accurate Coreference Resolution Defying Recent Trends*](https://aclanthology.org/2024.acl-long.722).  


# Python Package
The `maverick-coref` Python package provides an easy API to use Maverick models, enabling efficient and accurate coreference resolution with few lines of code.

Install the library from [PyPI](https://pypi.org/project/maverick-coref/)

```bash
pip install maverick-coref
```
or from source 

```bash
git clone https://github.com/SapienzaNLP/maverick-coref.git
cd maverick-coref
pip install -e .
```

## Loading a Pretrained Model
Maverick models can be loaded using huggingface_id or local path:
```bash
from maverick import Maverick
model = Maverick(
  hf_name_or_path = "maverick_hf_name" | "maverick_ckpt_path", default = "sapienzanlp/maverick-mes-ontonotes"
  device = "cpu" | "cuda", default = "cuda:0"
)
```

## Available Models

Available models at [SapienzaNLP huggingface hub](https://huggingface.co/collections/sapienzanlp/maverick-coreference-resolution-66a750a50246fad8d9c7086a):

|            hf_model_name            | training dataset | Score | Singletons |
|:-----------------------------------:|:----------------:|:-----:|:----------:|
|    ["sapienzanlp/maverick-mes-ontonotes"](https://huggingface.co/sapienzanlp/maverick-mes-ontonotes)    |     OntoNotes    |  83.6 |     No     |
|     ["sapienzanlp/maverick-mes-litbank"](https://huggingface.co/sapienzanlp/maverick-mes-litbank)     |      LitBank     |  78.0 |     Yes    |
|      ["sapienzanlp/maverick-mes-preco"](https://huggingface.co/sapienzanlp/maverick-mes-preco)      |       PreCo      |  87.4 |     Yes    |
<!-- |    "sapienzanlp/maverick-s2e-ontonotes"    |     OntoNotes    |  83.4 |     No     |     No    | -->
<!-- |    "sapienzanlp/maverick-incr-ontonotes"   |     Ontonotes    |  83.5 |     No     |     No    | -->
<!-- |  "sapienzanlp/maverick-mes-ontonotes-base" |     Ontonotes    |  81.4 |     No     |     No    | -->
<!-- | "sapienzanlp/maverick-s2e-ontonotes-base"  |     Ontonotes    |  81.1 |     No     |     No    | -->
<!-- | "sapienzanlp/maverick-incr-ontonotes-base" |     Ontonotes    |  81.0 |     No     |     No    | -->
<!-- |     "sapienzanlp/maverick-s2e-litbank"     |      LitBank     |  77.6 |     Yes    |     No    | -->
<!-- |     "sapienzanlp/maverick-incr-litbank"    |      LitBank     |  78.3 |     Yes    |     No    | -->
<!-- |      "sapienzanlp/maverick-s2e-preco"      |       PreCo      |  87.2 |     Yes    |     No    | -->
<!-- |      "sapienzanlp/maverick-incr-preco"     |       PreCo      |  88.0 |     Yes    |     No    | -->
N.B. Each dataset has different annotation guidelines, choose your model according to your use case.

## Inference
### Inputs
Maverick inputs can be formatted as either:
- plain text:
  ```bash
  text = "Barack Obama is traveling to Rome. The city is sunny and the president plans to visit its most important attractions"
  ```
- word-tokenized text, as a list of tokens:
  ```bash
  word_tokenized = ['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.',  'The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'attractions']
  ```
- sentence split, word-tokenized text, i.e., OntoNotes like input, as a list of lists of tokens:
  ```bash
  ontonotes_format = [['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.'], ['The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'attractions']] 
  ```

### Predict
You can use model.predict() to obtain coreference predictions.
For a sample input, the model will a dictionary containing:
- `tokens`, word tokenized version of the input.
- `clusters_token_offsets`, a list of clusters containing mentions' token offsets.
- `clusters_text_mentions`, a list of clusters containing mentions in plain text.

Example:
  ```bash
model.predict(ontonotes_format)
>>> {
  'tokens': ['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.', 'The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'monument', ',', 'the', 'Colosseum'], 
  'clusters_token_offsets': [[(5, 5), (7, 8), (17, 17)], [(0, 1), (12, 13)]], 
  'clusters_text_mentions': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]
}
```

If you input plain text, the model will include also char level offsets as `clusters_char_offsets`:
```bash
model.predict(text)
>>> {
  'tokens': [...], 
  'clusters_token_offsets': [...], 
  'clusters_char_offsets': [[(29, 32), (35, 42), (86, 88)], [(0, 11), (57, 69)]], 
  'clusters_text_mentions': [...]
  }
```

### 🚨Additional Features🚨
Since Coreference Resolution may serve as a stepping stone for many downstream use cases, in this package we cover multiple additional features:

- **Singletons**, either include or exclude singletons (i.e., single mention clusters) prediction by setting `singletons` to `True` or `False`.
*(hint: for accurate singletons use preco- or litbank-based models, since ontonotes does not include singletons and therefore the model is not trained to extract any)*
  ```bash
  #supported input: ontonotes_format
  model.predict(ontonotes_format, singletons=True)
  {'tokens': [...], 
  'clusters_token_offsets': [((5, 5), (7, 8), (17, 17)), ((0, 1), (12, 13)), ((17, 20),)],
  'clusters_char_offsets': None, 
  'clusters_token_text': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president'], ['its most important attractions']], 
  'clusters_char_text': None
  }
  ```

- **Clustering-only**, predict with predefined mentions (clustering-only), by passing mentions as a list of token offsets.
  ```bash
  #supported input: ontonotes_format
  mentions = [(0, 1), (5, 5), (7, 8)]
  model.predict(ontonotes_format, predefined_mentions=mentions)
  >>> {'tokens': [...], 
  'clusters_token_offsets': [((5, 5), (7, 8))],
  'clusters_char_offsets': None, 
  'clusters_token_text': [['Rome', 'The city']], 
  'clusters_char_text': None}
  ```

- **Starting from gold clusters**, predict starting from gold clusters, by passing the model the mentions as a list of token offsets.
*(Note: since starting clusters will be the first in the token offset outputs, to obtain the coreference resolution predictions **only for starting clusters** it is enough to take the first N clusters, where N is the number of starting clusters.)*
  ```bash
  #supported input: ontonotes_format 
  clusters = [[(5, 5), (7, 8)], [(0, 1)]]
  model.predict(ontonotes_format, add_gold_clusters=clusters)
  >>> {'tokens': [...], 'clusters_token_offsets': [((5, 5), (7, 8), (17, 17)), ((0, 1), (12, 13))], 'clusters_char_offsets': None, 'clusters_token_text': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']], 'clusters_char_text': None}
  ```

- **Speaker information**, since OntoNotes models are trained with additional speaker information [(more info here)](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf), you can specify speaker information with OntoNotes format. 

```bash
  #supported input: ontonotes_format 
  speakers = [["Mark", "Mark", "Mark", "Mark", "Mark"],["Jhon", "Jhon", "Jhon", "Jhon"]]
  model.predict(ontonotes_format, speakers=clusters)
```

# Using the official Training and Evaluation Script

This same repository contains also the code to train and evaluate Maverick systems using pytorch-lightning and Hydra.

**We strongly suggest to directly use the [python package](https://pypi.org/project/maverick-coref/) for easier inference and downstream usage.** 



## Environment
To set up the training and evaluation environment, run the bash script setup.sh that you can find at top level in this repository. This script will handle the creation of a new conda environment and will take care of all the requirements and data preprocessing for training and evaluating a model on OntoNotes. 

Simply run on the command line:
```
bash ./setup.sh
```
N.B. Remember to put the zip file *ontonotes-release-5.0_LDC2013T19.tgz* in the folder *data/prepare_ontonotes/* if you want to preprocess Ontonotes with the standard preprocessing proposed by [e2e-coref](https://github.com/kentonl/e2e-coref/). OntoNotes can be downloaded, upon registration, at the following [link](https://catalog.ldc.upenn.edu/LDC2013T19)

## Data 
Official Links:
- [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19)
- [PreCo](https://drive.google.com/file/d/1q0oMt1Ynitsww9GkuhuwNZNq6SjByu-Y/view)
- [LitBank](https://github.com/dbamman/litbank/tree/master/coref/conll)
- [WikiCoref](http://rali.iro.umontreal.ca/rali/?q=en/wikicoref)

Since those datasets usually require a preprocessing step to obtain the OntoNotes-like jsonlines format, we release ready-to-use version:
https://drive.google.com/drive/u/3/folders/18dtd1Qt4h7vezlm2G0hF72aqFcAEFCUo.


## Hydra
This repository uses [Hydra](https://hydra.cc/) configuration environment.

- In *conf/data/* each yaml file contains a dataset configuration.
- *conf/evaluation/* contains the model checkpoint file path and device settings for model evaluation.
- *conf/logging/* contains details for wandb logging.
- In *conf/model/*, each yaml file contains a model setup.
-  *conf/train/* contains training configurations.
- *conf/root.yaml* regulates the overall configuration of the environment.


## Train
To train a Maverick model, modify *conf/root.yaml* with your custom setup. 
By default, this file contains the settings for training and evaluating on the OntoNotes dataset.

To train a new model, follow the steps in  [Environment](#environment) section and run the following script:
```
conda activate maverick_env
python maverick/train.py
```


## Evaluate
To evaluate an existing model, it is necessary to set up two different environment variables.
1. Set the dataset path in conf/root.yaml, by default it is set to OntoNotes.
2. Set the model checkpoint path in conf/evaluation/default_evaluation.yaml.

Finally run the following:
```
conda activate env_name
python maverick/evaluate.py
```
This will directly output the CoNLL-2012 scores, and, under the experiments/ folder,  a output.jsonlines file containing the model outputs in OntoNotes style.

### Replicate paper results
The weights of each model can be found in the [SapienzaNLP huggingface hub](https://huggingface.co/collections/sapienzanlp/maverick-coreference-resolution-66a750a50246fad8d9c7086a).
To replicate any of the paper results,  download the weights.ckpt of a model from the its model card files and follow the steps reported in the [Evaluate](#evaluate) section.

E.G. to replicate the state of the art results of *Maverick_mes* on OntoNotes:
- download the weights from [here](https://huggingface.co/sapienzanlp/maverick-mes-ontonotes/blob/main/weights.ckpt).
- copy the local path of the weights in conf/evaluation/default_evaluation.yaml.
- activate the project's conda environment with *conda activate maverick_coref*.
- run *python maverick/evaluate.py*

# Citation
This work has been published at [ACL 2024 main conference](https://aclanthology.org/2024.acl-long.722.pdf). If you use any part, please consider citing our paper as follows:
```bibtex
@inproceedings{martinelli-etal-2024-maverick,
    title = "Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends",
    author = "Martinelli, Giuliano  and
      Barba, Edoardo  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.722",
    pages = "13380--13394",
}
```


## License

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).



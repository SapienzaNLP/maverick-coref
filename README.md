<h1 align="center">
  🤘Maverick Coref🤘
</h1>
<div align="center">


[![Conference](https://img.shields.io/badge/ACL%202024-red)](https://2024.aclweb.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Pip Package](https://img.shields.io/badge/🐍%20Python%20package-blue)](https://huggingface.co/Babelscape/cner-base)
</div>


This is the official repository for [*Maverick:
Efficient and Accurate Coreference Resolution Defying Recent Trends*](https://arxiv.org/pdf/todo/).  


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
  hf_name_or_path = "maverick_hf_name" | "maverick_ckpt_path", default = "SapienzaNLP/maverick-mes-ontonotes"
  device = "cpu" | "cuda", default = "cuda:0"
)
```

## Available Models
Available models at [SapienzaNLP huggingface hub](https://huggingface.co/sapienzanlp):

|            hf_model_name            | training dataset | Score | Singletons |
|:-----------------------------------:|:----------------:|:-----:|:----------:|
|    ["sapienzanlp/maverick-mes-ontonotes"](https://huggingface.co/sapienzanlp/maverick-mes-ontonotes)    |     OntoNotes    |  83,6 |     No     |
|     ["sapienzanlp/maverick-mes-litbank"](https://huggingface.co/sapienzanlp/maverick-mes-litbank)     |      LitBank     |  78,0 |     Yes    |
|      ["sapienzanlp/maverick-mes-preco"](https://huggingface.co/sapienzanlp/maverick-mes-preco)      |       PreCo      |  87,4 |     Yes    |
<!-- |    "sapienzanlp/maverick-s2e-ontonotes"    |     OntoNotes    |  83,4 |     No     |     No    | -->
<!-- |    "sapienzanlp/maverick-incr-ontonotes"   |     Ontonotes    |  83,5 |     No     |     No    | -->
<!-- |  "sapienzanlp/maverick-mes-ontonotes-base" |     Ontonotes    |  81,4 |     No     |     No    | -->
<!-- | "sapienzanlp/maverick-s2e-ontonotes-base"  |     Ontonotes    |  81,1 |     No     |     No    | -->
<!-- | "sapienzanlp/maverick-incr-ontonotes-base" |     Ontonotes    |  81,0 |     No     |     No    | -->
<!-- |     ["sapienzanlp/maverick-mes-litbank"](https://huggingface.co/sapienzanlp/maverick-mes-litbank)     |      LitBank     |  78,0 |     Yes    |    Yes    | -->
<!-- |     "sapienzanlp/maverick-s2e-litbank"     |      LitBank     |  77,6 |     Yes    |     No    | -->
<!-- |     "sapienzanlp/maverick-incr-litbank"    |      LitBank     |  78,3 |     Yes    |     No    | -->
<!-- |      ["sapienzanlp/maverick-mes-preco"](https://huggingface.co/sapienzanlp/maverick-mes-preco)      |       PreCo      |  87,4 |     Yes    |    Yes    | -->
<!-- |      "sapienzanlp/maverick-s2e-preco"      |       PreCo      |  87,2 |     Yes    |     No    | -->
<!-- |      "sapienzanlp/maverick-incr-preco"     |       PreCo      |  88,0 |     Yes    |     No    | -->


## Inference
### Inputs
Maverick inputs can be formatted as either 
- plain text:
  ```bash
  text = "Barack Obama is traveling to Rome. Today the city is sunny, therefore the president plans to visit its most important attraction, the Colosseum"
  ```
- word-tokenized text, as a list of tokens:
  ```bash
  word_tokenized = ['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.', 'The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'monument', ',', 'the', 'Colosseum']
  ```
- sentence splitted, word-tokenized text, i.e., OntoNotes like input, as a list of lists of tokens:
  ```bash
  ontonotes_format = [['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.'], ['The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'monument', ',', 'the', 'Colosseum']] 
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
  {'tokens': [...], 'clusters_token_offsets': [((17, 5)), ((5, 5), (7, 8), (17, 17)), ((0, 1), (12, 13))], 'clusters_char_offsets': None, 'clusters_token_text': [['its most important monument'],['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]}
  ```

- **Clustering-only**, predict with predefined mentions (clustering-only), by passing mentions as a list of token offsets.
  ```bash
  #supported input: ontonotes_format
  mentions = [(0, 1), (5, 5), (7, 8)]
  model.predict(ontonotes_format, predefine_mentions=mentions)
  >>> {'tokens': [...], 'clusters_token_offsets': [((5, 5), (7, 8), (17, 17)), ((0, 1), (12, 13))], 'clusters_char_offsets': None, 'clusters_token_text': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]}
  ```

- **Starting from gold clusters**, predict starting from gold clusters, by passing the model the mentions as a list of token offsets.
*(Note: since starting clusters will be the first in the token offset outputs, to obtain the coreference resolution predictions **only for starting clusters** it is enough to take the first N clusters, where N is the number of starting clusters.)*
  ```bash
  #supported input: ontonotes_format 
  clusters = [[(5, 5), (7, 8)], [(0, 1)]]
  model.predict(ontonotes_format, add_gold_clusters=clusters)
  >>> {'tokens': [...], 'clusters_token_offsets': [((5, 5), (7, 8), (17, 17)), ((0, 1), (12, 13))], 'clusters_char_offsets': None, 'clusters_token_text': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]}
  ```

- **Speaker information**, Since OntoNotes models are trained with additional speaker information [(more info here)](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf), you can specify speaker information with OntoNotes format. 

  ```bash
  #supported input: ontonotes_format 
  speakers = [["Mark", "Mark", "Mark", "Mark", "Mark"],["Jhon", "Jhon", "Jhon", "Jhon"]]
  model.predict(ontonotes_format, speakers=clusters)
```


# Official Training and Evaluation Script
This repository contains also the code to train and evaluate Maverick systems using pytorch-lightning and Hydra.
**We strongly suggest to use the [python package](http://pip.com) for easier inference** use the modelswhen using a to To train new Maverick systems, for re it is possible to use 
To set up the python environment for this project, we strongly suggest using the bash script setup.sh that you can find at top level in this repo. This script will create a new conda environment and take care of all the requirements and the data needed for the project. Simply run on the command line:
## Download Maverick Models
link to maverick pretrained models:
https://drive.google.com/drive/u/2/folders/1UXq4gWt1xYw2o1KDKhCtDsk5q0EiPx1t

All models can be found on [huggingface](https://huggingface.co/g185)

Put the zip file *ontonotes-release-5.0_LDC2013T19.tgz* in the folder *data/prepareontonotes/* if you want to preprocess Ontonotes, and then run 

```bash
git clone https://github.com/g185/maverick-coref.git
cd maverick-coref
bash ./setup.sh
``` 


# Environment Setup
To set up the python environment for this project, we strongly suggest using the bash script setup.sh that you can find at top level in this repo. This script will create a new conda environment and take care of all the requirements and the data needed for the project. Simply run on the command line:

```
bash ./setup.sh
```
Remember to put the zip file *ontonotes-release-5.0_LDC2013T19.tgz* in the folder *data/prepareontonotes/* if you want to preprocess Ontonotes with the standard preprocessing proposed by [e2e-coref](https://github.com/kentonl/e2e-coref/).

todo: add info about official scorer in https://github.com/conll/reference-coreference-scorers
bring experiments 


## Citation
This work has been published at NAACL 2024 (main conference). If you use any part, please consider citing our paper as follows:
```bibtex
@inproceedings{martinelli-etal-2024-cner,
    title = "CNER: Concept and Named Entity Recognition",
    author = "Martinelli, Giuliano and
      Molfese, Francesco  and
      Tedeschi, Simone  and
      Fernàndez-Castro, Alberte  and
      Navigli, Roberto",}
```
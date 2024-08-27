import pandas as pd
import spacy
import hydra
import torch
import pytorch_lightning as pl

from nltk import sent_tokenize
from torch.nn import Module, Linear, LayerNorm, Dropout, GELU

from maverick.common.constants import *


def get_category_id(mention, antecedent):
    mention, mention_pronoun_id = mention
    antecedent, antecedent_pronoun_id = antecedent

    if mention_pronoun_id > -1 and antecedent_pronoun_id > -1:
        if mention_pronoun_id == antecedent_pronoun_id:
            return CATEGORIES["pron-pron-comp"]
        else:
            return CATEGORIES["pron-pron-no-comp"]

    if mention_pronoun_id > -1 or antecedent_pronoun_id > -1:
        return CATEGORIES["pron-ent"]

    if mention == antecedent:
        return CATEGORIES["match"]

    union = mention.union(antecedent)
    if len(union) == max(len(mention), len(antecedent)):
        return CATEGORIES["contain"]

    return CATEGORIES["other"]


def get_pronoun_id(span):
    if len(span) == 1:
        span = list(span)
        if span[0] in PRONOUNS_GROUPS:
            return PRONOUNS_GROUPS[span[0]]
    return -1


def flatten(l):
    return [item for sublist in l for item in sublist]


def ontonotes_to_dataframe(file_path):
    # read file
    df = pd.read_json(hydra.utils.get_original_cwd() + "/" + file_path, lines=True)
    # ontonotes is split into words and sentences, jin sentences
    if "sentences" in df.columns:
        df["tokens"] = df["sentences"].apply(lambda x: flatten(x))
    elif "text" in df.columns:  # te
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        texts = df["text"].tolist()
        df["sentences"] = [[[tok.text for tok in s] for s in nlp.pipe(s)] for s in [sent_tokenize(text) for text in texts]]

        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        df["tokens"] = [[tok.text for tok in doc] for doc in nlp.pipe(texts)]
    # compute end of sequence indices
    if "preco" in file_path:
        df["EOS_indices"] = df["tokens"].apply(lambda x: [i + 1 for i, token in enumerate(x) if token == "."])
    else:
        df["EOS_lengths"] = df["sentences"].apply(lambda x: [len(value) for value in x])
        df["EOS_indices"] = df["EOS_lengths"].apply(lambda x: [sum(x[0 : (i[0] + 1)]) for i in enumerate(x)])
    # add speakers
    if "speakers" in df.columns and "wkc" not in file_path and "q" not in file_path:
        df["speakers"] = df["speakers"].apply(lambda x: flatten(x))
    else:
        df["speakers"] = df["tokens"].apply(lambda x: ["-"] * len(x))

    if "clusters" in df.columns:
        df = df[["doc_key", "tokens", "speakers", "clusters", "EOS_indices"]]
    else:
        df = df[["doc_key", "tokens", "speakers", "EOS_indices"]]
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def extract_mentions_to_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    return mention_to_gold


def original_token_offsets(clusters, subtoken_map, new_token_map):
    return [
        tuple(
            [
                (
                    new_token_map[subtoken_map[start]],
                    new_token_map[subtoken_map[end]],
                )
                for start, end in cluster
                if subtoken_map[start] is not None
                and subtoken_map[end] is not None  # only happens first evals, model predicts <s> as mentions
                and new_token_map[subtoken_map[start]]
                is not None  # it happens very rarely that in some weidly formatted sentences the model predicts the speaker name as a possible mention
            ]
        )
        for cluster in clusters
    ]


def unpad_gold_clusters(gold_clusters):
    new_gold_clusters = []
    for batch in gold_clusters:
        new_gold_clusters = []
        for cluster in batch:
            new_cluster = []
            for span in cluster:
                if span[0].item() != -1:
                    new_cluster.append((span[0].item(), span[1].item()))
            if len(new_cluster) != 0:
                new_gold_clusters.append(tuple(new_cluster))
    return new_gold_clusters


class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_prob):
        super(FullyConnectedLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense1 = Linear(self.input_dim, hidden_size)
        self.dense = Linear(hidden_size, self.output_dim)
        self.layer_norm = LayerNorm(hidden_size)
        self.activation_func = GELU()
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense1(temp)
        temp = self.dropout(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dense(temp)
        return temp


class RepresentationLayer(torch.nn.Module):
    def __init__(self, type, input_dim, output_dim, hidden_dim, **kwargs) -> None:
        super(RepresentationLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lt = type
        if type == "Linear":
            self.layer = Linear(input_dim, output_dim)
        elif type == "FC":
            self.layer = FullyConnectedLayer(input_dim, output_dim, hidden_dim, dropout_prob=0.2)
        elif type == "LSTM-left":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
        elif type == "LSTM-right":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
        elif type == "LSTM-bidirectional":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim / 2, bidirectional=True)
        elif type == "Conv1d":
            self.layer = torch.nn.Conv1d(input_size=input_dim, hidden_size=output_dim, kernel_size=7, stride=1, padding=3)
            self.dropout = Dropout(0.2)

    def forward(self, inputs):
        if self.lt == "Linear":
            return self.layer(inputs)
        elif self.lt == "FC":
            return self.layer(inputs)
        elif self.lt == "LSTM-left":
            return self.layer(inputs)[0][: self.hidden_dim]
        elif self.lt == "LSTM-right":
            return self.layer(inputs)[0][self.hidden_dim :]
        elif self.lt == "LSTM-bidirectional":
            return self.layer(inputs)[0]
        elif self.lt == "Conv1d":
            return self.layer(self.dropout(inputs))


def download_load_spacy():
    try:
        import nltk

        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        # colab fix
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
        except:
            nltk.download("punkt")
    except:
        from spacy.cli import download
        import nltk

        nltk.download("punkt")
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
    return nlp

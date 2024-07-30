import numpy as np
import hydra.utils
import torch

from typing import Tuple
from typing import Dict, Union
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk
from datasets import Dataset as dt

import maverick.common.util as util


NULL_ID_FOR_COREF = -1


class OntonotesDataset(Dataset):
    def __init__(self, name: str, path: str, batch_size, processed_dataset_path, tokenizer, **kwargs):
        super().__init__()
        self.stage = name
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True, add_prefix_space=True)
        self.max_doc_len = kwargs.get("max_doc_len", None)
        special_tokens_dict = {"additional_special_tokens": ["[SPEAKER_START]", "[SPEAKER_END]"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        try:
            self.set = load_from_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path + "/")
        except:
            self.set = dt.from_pandas(util.ontonotes_to_dataframe(path))
            if self.stage == "train":
                if "clusters" not in self.set.column_names:
                    print("Training set has to include cluster information")
                # self.set = self.set.map(self.cut_document_to_length, batched=False)
                self.set = self.prepare_data(self.set)
            self.set = self.set.map(self.encode, batched=False)
            self.set = self.set.remove_columns(column_names=["speakers"])
            if self.stage != "test":
                self.set.save_to_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path + "/")

    def prepare_data(self, set):
        return set.filter(lambda x: len(self.tokenizer(x["tokens"])["input_ids"]) <= self.max_doc_len)

    def cut_document_to_length(self, set_element):
        encoded_text = self.tokenizer(set_element["tokens"], add_special_tokens=True, is_split_into_words=True)
        if len(encoded_text["input_ids"]) <= self.max_doc_len + 1:
            result = set_element
        else:
            last_index_input_id_in_sentence = encoded_text.token_to_word(self.max_doc_len)
            eos_indices = [end for end in set_element["EOS_indices"] if end < last_index_input_id_in_sentence]
            last_sentence_end = eos_indices[-3]
            result = {
                "doc_key": set_element["doc_key"],
                "tokens": set_element["tokens"][:last_sentence_end],
                "speakers": set_element["speakers"][:last_sentence_end],
                "EOS_indices": eos_indices[:-3],
            }
            new_clusters = []
            for cluster in set_element["clusters"]:
                new_cluster = []
                for span in cluster:
                    if span[1] < last_sentence_end:
                        new_cluster.append(span)
                if len(new_cluster) >= 1:
                    new_clusters.append(new_cluster)
            result["clusters"] = new_clusters
        return result

    # missing genre information
    def _tokenize(self, tokens, clusters, speakers, eos_indices):
        token_to_new_token_map = []  # len() = len(tokens), contains indices of original sequence to new sequence
        new_token_map = []  # len() = len(new_tokens), contains indices of new sequence
        new_tokens = []  # contains new tokens
        last_speaker = None

        for idx, (token, speaker) in enumerate(zip(tokens, speakers)):
            if last_speaker != speaker:
                new_tokens += ["[SPEAKER_START]", speaker, "[SPEAKER_END]"]
                new_token_map += [None, None, None]
                last_speaker = speaker
            token_to_new_token_map.append(len(new_tokens))
            new_token_map.append(idx)
            new_tokens.append(token)

        for cluster in clusters:
            for start, end in cluster:
                assert tokens[start : end + 1] == new_tokens[token_to_new_token_map[start] : token_to_new_token_map[end] + 1]

        encoded_text = self.tokenizer(new_tokens, add_special_tokens=True, is_split_into_words=True)
        clusters = [
            [
                (
                    encoded_text.word_to_tokens(token_to_new_token_map[start]).start,
                    encoded_text.word_to_tokens(token_to_new_token_map[end]).end - 1,
                )
                for start, end in cluster
            ]
            for cluster in clusters
        ]
        eos_indices = [
            encoded_text.word_to_tokens(token_to_new_token_map[eos - 1]).start
            for eos in eos_indices
            if encoded_text.word_to_tokens(token_to_new_token_map[eos - 1]) != None
        ]
        output = {
            "tokens": tokens,
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "gold_clusters": clusters,
            "subtoken_map": encoded_text.word_ids(),
            "new_token_map": new_token_map,
            "EOS_indices": eos_indices,
        }
        return output

    def encode(self, example):
        if "clusters" not in example:
            example["clusters"] = []
        encoded = self._tokenize(
            example["tokens"],
            example["clusters"],
            example["speakers"],
            example["EOS_indices"],
        )  # debug when no clusters
        encoded["num_clusters"] = len(encoded["gold_clusters"]) if encoded["gold_clusters"] else 0
        encoded["max_cluster_size"] = max(len(c) for c in encoded["gold_clusters"]) if encoded["gold_clusters"] else 0
        encoded["length"] = len(encoded["input_ids"])
        return encoded

    def __len__(self) -> int:
        return self.set.shape[0]

    def __getitem__(self, index) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.set[index]

    # takes length of sequence (int) and eos_indices ([])
    # returns len x len zeros matrix with 1 in pos (start, all possible ends)
    def eos_mask(self, input_ids_len, eos_indices):
        mask = np.zeros((input_ids_len, input_ids_len))
        prec = 0
        for eos_idx in eos_indices:
            for i in range(prec, eos_idx + 2):
                for j in range(prec, eos_idx + 2):
                    mask[i][j] = 1
            prec = eos_idx
        mask = np.triu(mask)
        return mask

    # takes length of sequence (int) and coreferences ([[()]])
    # returns len x len zeros matrix with 1 in pos (start, end)
    def create_mention_matrix(self, input_ids_len, coreferences):
        matrix = np.zeros((input_ids_len, input_ids_len))
        for cluster in coreferences:
            for start_bpe_idx, end_bpe_idx in cluster:
                matrix[start_bpe_idx][end_bpe_idx] = 1
        return matrix

    # takes length of sequence (int) and coreferences ([[()]])
    # returns len zeros matrix with 1 in start position
    def create_start_matrix(self, input_ids_len, coreferences):
        matrix = np.zeros((input_ids_len))
        for cluster in coreferences:
            for start_bpe_idx, end_bpe_idx in cluster:
                matrix[start_bpe_idx] = 1
        return matrix

    # pad don't pad the rest, and is slow, think about something else
    def collate_fn(self, batch):
        batch = self.tokenizer.pad(batch)
        output = {
            "input_ids": torch.tensor(batch["input_ids"]),
            "attention_mask": torch.tensor(batch["attention_mask"]),
            "eos_mask": torch.tensor(self.eos_mask(len(batch["input_ids"][0]), batch["EOS_indices"][0])).unsqueeze(0),
            "gold_mentions": torch.tensor(
                self.create_mention_matrix(len(batch["input_ids"][0]), batch["gold_clusters"][0])
            ).unsqueeze(0),
            "gold_starts": torch.tensor(
                self.create_start_matrix(len(batch["input_ids"][0]), batch["gold_clusters"][0])
            ).unsqueeze(0),
        }

        max_num_clusters, max_max_cluster_size = max(batch["num_clusters"]), max(batch["max_cluster_size"])
        if max_num_clusters == 0:
            padded_clusters = []
        else:
            padded_clusters = [
                pad_clusters(cluster, max_num_clusters, max_max_cluster_size) for cluster in batch["gold_clusters"]
            ]

        output["gold_clusters"] = torch.tensor(padded_clusters)
        output["tokens"] = batch["tokens"]
        output["doc_key"] = batch["doc_key"]
        output["subtoken_map"] = batch["subtoken_map"]
        output["new_token_map"] = batch["new_token_map"]
        output["eos_indices"] = torch.tensor(batch["EOS_indices"])
        output["singletons"] = "litbank" in self.path or "preco" in self.path
        return output


def pad_clusters_inside(clusters, max_cluster_size):
    return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster)) for cluster in clusters]


def pad_clusters_outside(clusters, max_num_clusters):
    return clusters + [[]] * (max_num_clusters - len(clusters))


def pad_clusters(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside(clusters, max_cluster_size)
    return clusters

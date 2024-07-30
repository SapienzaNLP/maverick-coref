import torch
import numpy as np
import random

from transformers import (
    AutoModel,
    AutoConfig,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)

from maverick.common.util import *
from maverick.common.constants import *


class MentionClusterClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mention_hidden_states, cluster_hidden_states, attention_mask, labels=None):
        # repreated tensor of mention_hs, to append in first position for each possible mention cluster pair
        repeated_mention_hs = mention_hidden_states.unsqueeze(0).repeat(cluster_hidden_states.shape[0], 1, 1)

        # mention cluste pairs by contatenating mention vectors to cluster padded matrix
        mention_cluster_pairs = torch.cat((repeated_mention_hs, cluster_hidden_states), dim=1)
        attention_mask = torch.cat(
            (
                torch.ones(cluster_hidden_states.shape[0], 1, device=self.model.device),
                attention_mask,
            ),
            dim=1,
        )

        logits = self.model(inputs_embeds=mention_cluster_pairs, attention_mask=attention_mask).logits

        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1).to(self.model.device))
        return loss, logits


class Maverick_incr(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # document transformer encoder
        self.encoder_hf_model_name = kwargs["huggingface_model_name"]
        self.encoder = AutoModel.from_pretrained(self.encoder_hf_model_name)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_hf_model_name)
        self.encoder.resize_token_embeddings(self.encoder.embeddings.word_embeddings.num_embeddings + 3)

        # freeze
        if kwargs["freeze_encoder"]:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # type of representation layer in 'Linear, FC, LSTM-left, LSTM-right, Conv1d'
        self.representation_layer_type = "FC"  # fullyconnected
        # span hidden dimension
        self.token_hidden_size = self.encoder_config.hidden_size

        # if span representation method is to concatenate start and end, a mention hidden size will be 2*token_hidden_size
        self.mention_hidden_size = self.token_hidden_size * 2

        # incremental transformer classifier
        self.incremental_model_hidden_size = kwargs.get("incremental_model_hidden_size", 384)  # 768/2
        self.incremental_model_num_layers = kwargs.get("incremental_model_num_layers", 1)
        self.incremental_model_config = DistilBertConfig(num_labels=1, hidden_size=self.incremental_model_hidden_size)
        self.incremental_model = DistilBertForSequenceClassification(self.incremental_model_config).to(self.encoder.device)
        self.incremental_model.distilbert.transformer.layer = self.incremental_model.distilbert.transformer.layer[
            : self.incremental_model_num_layers
        ]
        self.incremental_model.distilbert.embeddings.word_embeddings = None
        self.incremental_transformer = MentionClusterClassifier(model=self.incremental_model)

        # encodes mentions for incremental clustering
        self.incremental_span_encoder = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.mention_hidden_size,
            output_dim=self.incremental_model_hidden_size,
            hidden_dim=int(self.mention_hidden_size / 2),
        )

        # mention extraction layers
        # representation of start token
        self.start_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        # representation of end token
        self.end_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        # models probability to be the start of a mention
        self.start_token_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
        )

        # model mention probability from start and end representations
        self.start_end_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.mention_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
        )

    # takes last_hidden_states, eos_mask, ground truth and stage
    def squad_mention_extraction(self, lhs, eos_mask, gold_mentions, gold_starts, stage):
        start_idxs = []
        mention_idxs = []
        start_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        mention_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)

        for bidx in range(0, lhs.shape[0]):
            lhs_batch = lhs[bidx]  # SEQ_LEN X HIDD_DIM
            eos_mask_batch = eos_mask[bidx]  # SEQ_LEN X SEQ_LEN

            # compute start logits
            start_logits_batch = self.start_token_classifier(lhs_batch).squeeze(-1)  # SEQ_LEN

            if gold_starts != None:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(start_logits_batch, gold_starts[bidx])

                # accumulate loss
                start_loss = start_loss + loss

            # compute start positions
            start_idxs_batch = ((torch.sigmoid(start_logits_batch) > 0.5)).nonzero(as_tuple=False).squeeze(-1)

            start_idxs.append(start_idxs_batch.detach().clone())
            # in training, use gold starts to learn to extract mentions, inference use predicted ones
            if stage == "train":
                start_idxs_batch = (
                    ((torch.sigmoid(gold_starts[bidx]) > 0.5)).nonzero(as_tuple=False).squeeze(-1)
                )  # NUM_GOLD_STARTS

            # contains all possible start end indices pairs, i.e. for all starts, all possible ends looking at EOS index
            possibles_start_end_idxs = (eos_mask_batch[start_idxs_batch] == 1).nonzero(as_tuple=False)  # STARTS x 2

            # this is to have reference respect to original positions
            possibles_start_end_idxs[:, 0] = start_idxs_batch[possibles_start_end_idxs[:, 0]]

            possible_start_idxs = possibles_start_end_idxs[:, 0]
            possible_end_idxs = possibles_start_end_idxs[:, 1]

            # extract start and end hidden states
            starts_hidden_states = lhs_batch[possible_end_idxs]  # start
            ends_hidden_states = lhs_batch[possible_start_idxs]  # end

            # concatenation of start to end representations created using a representation layer
            s2e_representations = torch.cat(
                (
                    self.start_token_representation(starts_hidden_states),
                    self.end_token_representation(ends_hidden_states),
                ),
                dim=-1,
            )

            # classification of mentions
            s2e_logits = self.start_end_classifier(s2e_representations).squeeze(-1)

            # mention_start_idxs and mention_end_idxs
            mention_idxs.append(possibles_start_end_idxs[torch.sigmoid(s2e_logits) > 0.5].detach().clone())

            if s2e_logits.shape[0] != 0:
                if gold_mentions != None:
                    mention_loss_batch = torch.nn.functional.binary_cross_entropy_with_logits(
                        s2e_logits,
                        gold_mentions[bidx][possible_start_idxs, possible_end_idxs],
                    )
                    mention_loss = mention_loss + mention_loss_batch

        return (start_idxs, mention_idxs, start_loss, mention_loss)

    def incremental_span_clustering(self, mentions_hidden_states, mentions_idxs, gold_clusters, stage):
        pred_cluster_idxs = []  # cluster_idxs = list of list of tuple of offsets (also output) up to mention_idx
        if gold_clusters != None:
            gold_cluster_idxs = unpad_gold_clusters(gold_clusters)  # gold_cluster_idxs, but padded

        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.incremental_model.device)
        mentions_hidden_states = mentions_hidden_states[0]
        idx_to_hs = dict(zip([tuple(m) for m in mentions_idxs.tolist()], mentions_hidden_states))

        # for each mention
        for idx, (
            mention_hidden_states,
            (mention_start_idx, mention_end_idx),
        ) in enumerate(zip(mentions_hidden_states, mentions_idxs)):
            if idx == 0:
                # if first create singleton cluster
                pred_cluster_idxs.append([(mention_start_idx.item(), mention_end_idx.item())])
            else:
                if stage == "train":
                    # if we are in training, retrieve use gold cluster idx to induce loss.
                    cluster_idx, labels = self.new_cluster_idxs_labels(
                        (mention_start_idx, mention_end_idx), gold_cluster_idxs
                    )  # can be used using only tensors
                else:
                    cluster_idx, labels = pred_cluster_idxs, None

                # get cluster padded matrix matrix and attention mask (excludes padding)
                cluster_hs, cluster_am = self.get_cluster_states_matrix(idx_to_hs, cluster_idx, stage)

                # produce logits for each possible cluster mention pair
                mention_cluster_loss, logits = self.incremental_transformer(
                    mention_hidden_states=mention_hidden_states,
                    cluster_hidden_states=cluster_hs,
                    attention_mask=cluster_am,
                    labels=labels,
                )

                if mention_cluster_loss != None:
                    coreference_loss = coreference_loss + mention_cluster_loss

                if stage != "train":
                    # only in inference
                    num_possible_clustering = torch.sum(torch.sigmoid(logits) > 0.5, dim=0).bool().float()

                    if num_possible_clustering == 0:
                        # if no clustering, create new singleton cluster
                        pred_cluster_idxs.append([(mention_start_idx.item(), mention_end_idx.item())])
                    else:
                        # otherwise, take most probabile clustering predicted by the model and assign this mention to that cluster
                        assigned_idx = logits.argmax(axis=0).detach().cpu()
                        pred_cluster_idxs[assigned_idx.item()].append((mention_start_idx.item(), mention_end_idx.item()))
        # normalize loss debug
        if gold_clusters != None:
            coreference_loss = coreference_loss / (mentions_hidden_states.shape[0] if mentions_hidden_states.shape[0] != 0 else 1)

            # coreference_loss = coreference_loss / (len(gold_cluster_idxs) if len(gold_cluster_idxs) != 0 else 1)
        coreferences_pred = [tuple(item) for item in pred_cluster_idxs]  # if len(item) > 1]
        return coreference_loss, coreferences_pred

    def get_cluster_states_matrix(self, idx_to_hs, cluster_idxs, stage):
        # create padded matrix of encoded mentions
        max_length = max([len(x) for x in cluster_idxs])
        if stage == "train":
            max_length = max_length if max_length < 31 else 30
        forward_matrix = torch.zeros(
            (len(cluster_idxs), max_length, self.incremental_model_hidden_size),
            device=self.encoder.device,
        )
        forward_am = torch.zeros((len(cluster_idxs), max_length), device=self.encoder.device)

        for cluster_idx, span_idxs in enumerate(cluster_idxs):
            if stage == "train":
                if len(span_idxs) > 30:
                    span_idxs = sorted(span_idxs)
                    new_idxs = [span_idxs[0]]
                    new_idxs.extend(random.sample(span_idxs, 28))
                    new_idxs.append(span_idxs[-1])
                    span_idxs = new_idxs

            hs = torch.stack([idx_to_hs[span_idx] for span_idx in span_idxs])

            forward_matrix[cluster_idx][: hs.shape[0]] = hs
            forward_am[cluster_idx][: hs.shape[0]] = torch.ones((hs.shape[0]), device=self.encoder.device)

        return forward_matrix, forward_am

    # takes the index of the mention (mention_start, mention_end) and gold coreferences, returns filtered indices (up to mention idx) and labels
    def new_cluster_idxs_labels(self, mention_idxs, gold_coreference_idxs):
        res_coreference_idxs = []
        # list of length number of clusters in gold, and 1.0 where the mention is laying
        labels = [
            1.0 if (mention_idxs[0].item(), mention_idxs[1].item()) in span_idx else 0.0 for span_idx in gold_coreference_idxs
        ]
        # filter cluster up to the mention you are evaluating
        for cluster_idxs in gold_coreference_idxs:
            idxs = []
            for span_idx in cluster_idxs:
                # if span is antecedent to current mention, stay in possible clusters
                if span_idx[0] < mention_idxs[0].item() or (
                    span_idx[0] == mention_idxs[0].item() and span_idx[1] < mention_idxs[1].item()
                ):
                    idxs.append((span_idx[0], span_idx[1]))
            # idxs = sorted(idxs, reverse=True)
            res_coreference_idxs.append(idxs)

        labels = torch.tensor(
            [lab for lab, idx in zip(labels, res_coreference_idxs) if len(idx) != 0],
            device=self.encoder.device,
        )
        res_coreference_idxs = [idx for idx in res_coreference_idxs if len(idx) != 0]
        return res_coreference_idxs, labels

    def forward(
        self,
        stage,
        input_ids,
        attention_mask,
        eos_mask,
        gold_starts=None,
        gold_mentions=None,
        gold_clusters=None,
    ):
        last_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # B x S x TH

        lhs = last_hidden_states

        loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        loss_dict = {}
        preds = {}

        (
            start_idxs,
            mention_idxs,
            start_loss,
            mention_loss,
        ) = self.squad_mention_extraction(
            lhs=last_hidden_states,
            eos_mask=eos_mask,
            gold_mentions=gold_mentions,
            gold_starts=gold_starts,
            stage=stage,
        )

        loss_dict["start_loss"] = start_loss
        preds["start_idxs"] = [start.detach().cpu() for start in start_idxs]

        loss_dict["mention_loss"] = mention_loss
        preds["mention_idxs"] = [mention.detach().cpu() for mention in mention_idxs]

        loss = loss + start_loss + mention_loss

        if stage == "train":
            mention_idxs = (gold_mentions[0] == 1).nonzero(as_tuple=False)
        else:
            mention_idxs = mention_idxs[0]

        mention_start_idxs = mention_idxs[:, 0]
        mention_end_idxs = mention_idxs[:, 1]

        mentions_start_hidden_states = torch.index_select(lhs, 1, mention_start_idxs)
        mentions_end_hidden_states = torch.index_select(lhs, 1, mention_end_idxs)

        mentions_hidden_states = torch.cat((mentions_start_hidden_states, mentions_end_hidden_states), dim=2)

        mentions_hidden_states = self.incremental_span_encoder(mentions_hidden_states)

        coreference_loss, coreferences = self.incremental_span_clustering(
            mentions_hidden_states, mention_idxs, gold_clusters, stage
        )

        loss = loss + coreference_loss
        loss_dict["coreference_loss"] = coreference_loss

        if stage != "train":
            preds["clusters"] = coreferences

        loss_dict["full_loss"] = loss
        output = {"pred_dict": preds, "loss_dict": loss_dict, "loss": loss}

        return output

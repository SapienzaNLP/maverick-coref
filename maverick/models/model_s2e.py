import torch
import numpy as np

from transformers import AutoModel, AutoConfig

from maverick.common.util import *
from maverick.common.constants import *


class Maverick_s2e(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # document transformer encoder
        self.encoder_hf_model_name = kwargs["huggingface_model_name"]
        self.encoder = AutoModel.from_pretrained(self.encoder_hf_model_name)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_hf_model_name)
        # self.encoder_config.attention_window = 1024
        self.encoder.resize_token_embeddings(self.encoder.embeddings.word_embeddings.num_embeddings + 3)

        # freeze
        if kwargs["freeze_encoder"]:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # span representation, now is concat_start_end
        self.span_representation = kwargs["span_representation"]
        # type of representation layer in 'Linear, FC, LSTM-left, LSTM-right, Conv1d'
        self.representation_layer_type = "FC"  # fullyconnected
        # span hidden dimension
        self.token_hidden_size = self.encoder_config.hidden_size

        # if span representation method is to concatenate start and end, a mention hidden size will be 2*token_hidden_size
        if self.span_representation == "concat_start_end":
            self.mention_hidden_size = self.token_hidden_size * 2

        self.antecedent_s2s_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )
        self.antecedent_e2e_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        self.antecedent_s2e_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )
        self.antecedent_e2s_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
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
    # check if mention_mask deletes some good gold mentions!
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

    def create_mention_to_antecedent_singletons(self, span_starts, span_ends, coref_logits):
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        bs, n_spans, _ = coref_logits.shape

        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        span_starts = span_starts.detach().cpu()
        span_ends = span_ends.detach().cpu()
        max_antecedents = coref_logits.argmax(axis=-1).detach().cpu()
        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

        antecedent_indices = max_antecedents[max_antecedents < n_spans]
        span_indices = np.stack([span_starts.detach().cpu(), span_ends.detach().cpu()], axis=-1)

        mentions = span_indices[doc_indices, mention_indices]
        antecedents = span_indices[doc_indices, antecedent_indices]
        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        sing_indices = np.zeros_like(len(np.setdiff1d(non_mentions, antecedent_indices)))
        singletons = span_indices[sing_indices, np.setdiff1d(non_mentions, antecedent_indices)]

        # mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1 and len(antecedents.shape) == 1:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=0)
        else:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1:
            mention_to_antecedent = [mention_to_antecedent]

        if len(singletons.shape) == 1:
            singletons = [singletons]

        return doc_indices, mention_to_antecedent, singletons

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k), device="cpu")
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(
            zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)
        ):
            gold_clusters = self.extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.encoder.device)
        return new_cluster_labels

    def s2e_mention_cluster(self, mention_start_reps, mention_end_reps, mention_start_idxs, mention_end_idxs, gold, stage):
        coref_logits = self._calc_coref_logits(mention_start_reps, mention_end_reps)
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)

        coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        if stage == "train":
            labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, gold)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, labels)

        doc, m2a, singletons = self.create_mention_to_antecedent_singletons(mention_start_idxs, mention_end_idxs, coref_logits)
        coreferences = self.create_clusters(m2a, singletons)
        return coreference_loss, coreferences

    def create_clusters(self, m2a, singletons):
        # Note: mention_to_antecedent is a numpy array

        clusters, mention_to_cluster = [], {}
        for mention, antecedent in m2a:
            mention, antecedent = tuple(mention), tuple(antecedent)
            if antecedent in mention_to_cluster:
                cluster_idx = mention_to_cluster[antecedent]
                if mention not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(mention)
                    mention_to_cluster[mention] = cluster_idx
            elif mention in mention_to_cluster:
                cluster_idx = mention_to_cluster[mention]
                if antecedent not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(antecedent)
                    mention_to_cluster[antecedent] = cluster_idx
            else:
                cluster_idx = len(clusters)
                mention_to_cluster[mention] = cluster_idx
                mention_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, mention])

        clusters = [tuple(cluster) for cluster in clusters]
        if len(singletons) != 0:
            clust = []
            while len(clusters) != 0 or len(singletons) != 0:
                if len(singletons) == 0:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
                elif len(clusters) == 0:
                    clust.append(tuple([tuple(singletons[0])]))
                    singletons = singletons[1:]
                elif singletons[0][0] < sorted(clusters[0], key=lambda x: x[0])[0][0]:
                    clust.append(tuple([tuple(singletons[0])]))
                    singletons = singletons[1:]
                else:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
            return clust
        return clusters

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s
        temp = self.antecedent_s2s_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = (
            top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
        )  # [batch_size, max_k, max_k]
        return coref_logits

    def extract_clusters(gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in cluster if (-1) not in m) for cluster in gold_clusters]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        return gold_clusters

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

        coreference_loss, coreferences = self.s2e_mention_cluster(
            mentions_start_hidden_states,
            mentions_end_hidden_states,
            mention_start_idxs,
            mention_end_idxs,
            gold_clusters,
            stage,
        )

        loss = loss + coreference_loss
        loss_dict["coreference_loss"] = coreference_loss

        if stage != "train":
            preds["clusters"] = coreferences

        loss_dict["full_loss"] = loss
        output = {"pred_dict": preds, "loss_dict": loss_dict, "loss": loss}

        return output

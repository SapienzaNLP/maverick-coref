import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


# Mention evaluation performed as scores on set of mentions.
class OfficialMentionEvaluator:
    def __init__(self):
        self.tp, self.fp, self.fn = 0, 0, 0

    # takes predicted mentions and gold mentions:
    # a mention is  [[(wstart, wend)]]
    # calculates tp, fp, and fn as set operations
    # returns prf using formula
    def update(self, predicted_mentions, gold_mentions):
        predicted_mentions = set(predicted_mentions)
        gold_mentions = set(gold_mentions)

        self.tp += len(predicted_mentions & gold_mentions)
        self.fp += len(predicted_mentions - gold_mentions)
        self.fn += len(gold_mentions - predicted_mentions)

    def get_f1(self):
        pr = self.get_precision()
        rec = self.get_recall()
        return 2 * pr * rec / (pr + rec) if pr + rec > 0 else 0.0

    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


# CoNLL-2012 metrics https://aclanthology.org/W12-4501.pdf#page=19
class OfficialCoNLL2012CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [OfficialEvaluator(m) for m in (muc, b_cubed, ceafe)]
        self.metrics = {"muc": 0, "b_cubed": 1, "ceafe": 2}

    # pred & golds : [[((wstart, wend),()), (())...]],  mention_to_pred and mention_to_gold: [{(wstart,wend):((wstart, wend)...)}]
    def update(self, pred, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(pred, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self, metric):
        if metric == "conll2012":
            return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)
        else:
            return self.evaluators[self.metrics[metric]].get_f1()

    def get_recall(self, metric):
        if metric == "conll2012":
            return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)
        else:
            return self.evaluators[self.metrics[metric]].get_recall()

    def get_precision(self, metric):
        if metric == "conll2012":
            return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)
        else:
            return self.evaluators[self.metrics[metric]].get_precision()

    def get_prf(self, metric):
        return self.get_precision(metric), self.get_recall(metric), self.get_f1(metric)


class OfficialEvaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


# mention based metric (Bagga and Baldwin, 1998)
def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count
        if len(c) != 0:
            num += correct / float(len(c))
            dem += len(c)

    return num, dem


# link based metric (Vilain et al. 2995).
def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


# Entity based metric (Luo, 2005)
def ceafe(clusters, gold_clusters):
    # clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = sum(scores[row_ind, col_ind])
    return similarity, len(clusters), similarity, len(gold_clusters)


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1 :]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

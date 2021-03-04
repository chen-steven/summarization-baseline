from collections import Counter
class ExtractionScorer:
    def __init__(self, ignore_idx=-1):
        self.ignore_idx = ignore_idx

    def _compute_f1(self, pred_chain, gt_chain):
        common = Counter(pred_chain) & Counter(gt_chain)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = num_same / len(pred_chain)
        recall = num_same / len(gt_chain)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def _set_match(self, pred, gt):
        return set([int(x) for x in pred]) == set([int(x) for x in gt])

    def _postprocess(self, gumbel_outputs, sentence_labels):
        predictions = [[idx for idx, x in enumerate(gumbel) if x == 1] for gumbel in gumbel_outputs]
        sentence_idxs = [[x for x in label if x != self.ignore_idx] for label in sentence_labels]
        return predictions, sentence_idxs

    def compute_metric(self, gumbel_outputs, sentence_labels, postprocess=True):
        d = Counter()
        total = 0
        if postprocess:
            predictions, sentence_idxs = self._postprocess(gumbel_outputs, sentence_labels)
        else:
            predictions = gumbel_outputs
            sentence_idxs = sentence_labels
       # predictions = [[idx for idx, x in enumerate(gumbel) if x == 1] for gumbel in gumbel_outputs]
        #sentence_idxs = [[x for x in label if x != self.ignore_idx] for label in sentence_labels]

        for pred, gt in zip(predictions, sentence_idxs):
            total += 1
            d['extraction_em'] += self._set_match(pred, gt)
            f1, p, r = self._compute_f1(pred, gt)
            d['extraction_f1'] += f1
            d['precision'] += p
            d['recall'] += r

        return {key: d[key] / total for key in d}



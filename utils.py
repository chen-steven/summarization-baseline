import torch

def convert_attention_mask(sentence_indicator, gumbel_output):
    batch_size, sent_num = gumbel_output.size()
    batch_idx = torch.range(0, batch_size - 1, dtype=torch.long).reshape(-1, 1).cuda()
    idx = batch_idx * (sent_num) + sentence_indicator
    return gumbel_output.reshape(-1)[idx]

def gumbel_softmax_topk(logits, k, tau=1, hard=False, dim=-1):
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    
    if hard:
        # Straight through.
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def convert_one_hot(tensor, size, pad_id=-1):
    one_hot = torch.zeros(tensor.size(0), size+1).cuda()
    max_val = size
    mask = (tensor == pad_id).long()
    src = tensor*(1-mask)+mask*max_val
#    one_hot = torch.zeros(tensor.size(0), size + 1 if includes_pad else size).cuda()
    one_hot = one_hot.scatter(1, src, 1)
    return one_hot[:, :-1]# if includes_pad else one_hot

def mask_sentences(logits, sentence_indicator, mask_val=-1e30):
    idxs = torch.arange(logits.size(1)).expand(sentence_indicator.size(0), -1).cuda()
    m, _ = sentence_indicator.max(-1)
    m = m.unsqueeze(-1)
    mask = (idxs < m).float().unsqueeze(-1)
    return logits*mask + (1-mask)*mask_val


def get_sentence_mask(sentence_indicator, num_sentences):
    idxs = torch.arange(num_sentences).expand(sentence_indicator.size(0), -1).cuda()
    m, _ = sentence_indicator.max(-1)
    m = m.unsqueeze(-1)
    return idxs < m
    

import torch

def convert_attention_mask(sentence_indicator, gumbel_output):
    batch_size, sent_num = gumbel_output.size()
    batch_idx = torch.range(0, batch_size - 1, dtype=torch.long).reshape(-1, 1).cuda()
    idx = batch_idx * (sent_num) + sentence_indicator
    return gumbel_output.view(-1)[idx]

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

def convert_one_hot(tensor, size):
    one_hot = torch.zeros(tensor.size(0), size)
    one_hot = one_hot.scatter(1, tensor, 1)
    return one_hot


import torch

def convert_attention_mask(sentence_indicator, gumbel_output):
    batch_size, sent_num = gumbel_output.size()
    batch_idx = torch.range(0, batch_size - 1, dtype=torch.long).reshape(-1, 1).cuda()
    idx = batch_idx * (sent_num) + sentence_indicator
    return gumbel_output.view(-1)[idx]

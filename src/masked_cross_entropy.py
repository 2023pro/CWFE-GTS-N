import torch
from torch.nn import functional
import copy
from src.tree import *
from src.pre_data import *

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def m_c_entropy(logits, target, length,O_L):
    #target[B x 表達式長度]

    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    negative_target = []  # 存储负标签
    negative_mask = []
    tar = copy.deepcopy(target)
    max_target_length = max(length)
    loss_length = 0 #算ne_loss的時候用到

    for tag, l in zip(tar, length):
        tag1 = copy.deepcopy(tag)
        temp = tag1[:l].detach().cpu().numpy()
        try:
            temp = buildtree(temp,O_L)
            temp = pad_seq(temp, l, max_target_length)
            negative_target.append(temp)
            mask_list = list(map(lambda x, y: 0 if x == y else 1, temp, tag))#temp和tag做對比，相同位置，值不一樣在mask_list置為1
            mask_list[0] = 0
            for i in range(max_target_length):
                if mask_list[i] ==1:
                    loss_length +=1
            negative_mask.append(mask_list)
        except:
            mask_list = [0] * max_target_length
            negative_mask.append(mask_list)
            negative_target.append(mask_list)  # 出錯時，隨便填東西進負標簽中

    if torch.cuda.is_available():
        neg_target = torch.LongTensor(negative_target).cuda()
        neg_mask = torch.LongTensor(negative_mask).cuda()
    else:
        neg_target = torch.LongTensor(negative_target)
        neg_mask = torch.LongTensor(negative_mask)


    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    ne_log_probs_flat = functional.softmax(logits_flat, dim=1)  # --------------------------

    # log_probs_flat: (batch * max_len, num_classes)
    #log_probs_flat = (1 - ne_log_probs_flat) ** 1 *  functional.log_softmax(logits_flat, dim=1)
    log_probs_flat =  functional.log_softmax(logits_flat, dim=1)

    # ne_log_probs_flat = torch.log(1. - torch.exp(log_probs_flat))#-----------------------
    ne_log_probs_flat = 1 - ne_log_probs_flat
    ne_log_probs_flat[ne_log_probs_flat <=  0.00001] = 0.00001
    ne_log_probs_flat = torch.log(ne_log_probs_flat)  #
    tryyy = ne_log_probs_flat#---------------------------------------------------------------

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    neg_target_flat = neg_target.view(-1,1)

    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    #ne_llosses_flat = torch.gather(log_probs_flat, dim=1, index=neg_target_flat)
    tryyy = -torch.gather(tryyy, dim=1, index=neg_target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    #ne_losses = ne_llosses_flat.view(*neg_target.size())
    tryyy = tryyy.view(*neg_target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()

    #ne_losses = ne_losses * neg_mask.float()
    #ne_loss = ne_losses.sum() / loss_length

    tryyy = tryyy * neg_mask.float()
    if loss_length == 0:
        loss_length += 1
    tryyy = tryyy.sum()/float(loss_length)


    loss = loss+tryyy

    return loss

def masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss





import torch
import torch.nn.functional as F

def MME_entropy(output_t1):
    output_t1 = F.softmax(output_t1, dim=1)
    lossU = -(-output_t1 * torch.log(output_t1 + 1e-6)).sum(1).mean()
    return lossU

def Entropy(output_t1,output_t2):
    output_t1 = F.softmax(output_t1,dim = 1)
    output_t2 = F.softmax(output_t2,dim = 1)
    entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
    entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
    return entropy_loss
import torch
import torch.nn.functional as F

def BCDM_Discrepancy(output_t1,output_t2):
    output_t1 = F.softmax(output_t1,dim = 1)
    output_t2 = F.softmax(output_t2,dim = 1)

    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss

def MCD_Discrepancy(output_t1,output_t2):
    output_t1 = F.softmax(output_t1,dim = 1)
    output_t2 = F.softmax(output_t2,dim = 1)
    return torch.mean(torch.abs(output_t1 - output_t2))


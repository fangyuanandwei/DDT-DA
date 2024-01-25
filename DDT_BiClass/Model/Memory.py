import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class Memory(nn.Module):
    def __init__(self, class_number, feature_dim, target_number, top_number):
        super(Memory, self).__init__()
        self.class_number  = class_number
        self.feature_dim   = feature_dim
        self.target_number = target_number
        self.top_number = top_number
        self.momentum = True
        
        if torch.cuda.is_available():
            self.target_featurememory = torch.zeros(self.target_number, self.feature_dim).cuda()
            self.center_featurememory = torch.zeros(self.class_number, self.feature_dim).cuda()

            self.target_softmaxFc1memory = torch.zeros(self.target_number, self.class_number).cuda()
            self.target_softmaxFc2memory = torch.zeros(self.target_number, self.class_number).cuda()


    def init_memory(self,model,target_dataloader):
        '''
        :param               model       : Generator
        :param   target_dataloader       :
        :return:self.target_featurememory:[target_number,feature_dim]
        '''
        model.eval()
        for batch_idx, (data, gt, index) in enumerate(target_dataloader):
            data = to_cuda(data)
            feature = model(data)
            self.target_featurememory[index] = feature.data

    def update(self, feature, fc1, fc2, index_target):
        if self.momentum:
            momentum = 0.1
            self.target_featurememory[index_target] = (1.0 - momentum) * self.target_featurememory[index_target] + momentum * feature.data
            self.target_softmaxFc1memory[index_target] = (1.0 - momentum) * self.target_softmaxFc1memory[index_target] + momentum * fc1.data
            self.target_softmaxFc2memory[index_target] = (1.0 - momentum) * self.target_softmaxFc2memory[index_target] + momentum * fc2.data
        else:
            self.target_featurememory[index_target,:] = feature.data
            self.target_softmaxFc1memory[index_target,:] = fc1.data
            self.target_softmaxFc2memory[index_target,:] = fc2.data



    def feature_smiliarity(self,target_feature,index_target):
        '''
        Calculate the softmax example with the most similar feature
        (计算在特征空间上最相似的样本的分类相似性,return[*,31])
        Feature Similarity
        target_feature                :[8,256]
        self.target_featurememory     :[498,256]
        FeatureDistance_matrix        :[8,498]
        FeatureMostSimilarity_distance:[8,3]
        FeatureMostSimilarity_index   :[8,3]
        Fc1memory:[16,31]        
        '''
        FeatureDistance_matrix = self.CosDistance(target_feature, self.target_featurememory)
        
        #排除第一个
        FeatureMostSimilarity_distance, FeatureMostSimilarity_index = torch.topk(FeatureDistance_matrix, self.top_number + 1, dim=1, largest=False)
        FeatureMostSimilarity_index_other = FeatureMostSimilarity_index[:, 1:]        
        FeatureMostSimilarity_index_flatten = torch.flatten(FeatureMostSimilarity_index_other)

        #不排除第一个
        #FeatureMostSimilarity_distance, FeatureMostSimilarity_index = torch.topk(FeatureDistance_matrix, self.top_number, dim=1, largest=False)
        #FeatureMostSimilarity_index_flatten = torch.flatten(FeatureMostSimilarity_index)

        Fc1memory = self.target_softmaxFc1memory[FeatureMostSimilarity_index_flatten, :]
        Fc2memory = self.target_softmaxFc2memory[FeatureMostSimilarity_index_flatten, :]

        return Fc1memory, Fc2memory

    def softmax_smiliarity(self, fc1, fc2, index_target):
        '''
        Calculate the feture example with the most similar classification
        (计算在分类输出概率上最相似样本在特征空间中特征,renturn[*,256])
        Fc1'''
        
        SoftmaxFc1Distance_matrix = self.CosDistance(fc1,self.target_softmaxFc1memory)
        Fc1MostSimilarity_distance, Fc1MostSimilarity_index = torch.topk(SoftmaxFc1Distance_matrix,self.top_number,dim=1,largest=False)
        Fc1MostSimilarity_index = Fc1MostSimilarity_index[:, 1:]
        Fc1MostSimilarity_index = torch.flatten(Fc1MostSimilarity_index)
        Featurememory1 = self.target_featurememory[Fc1MostSimilarity_index,:]

        '''Fc2'''
        SoftmaxFc2Distance_matrix = self.CosDistance(fc2,self.target_softmaxFc2memory)
        Fc2MostSimilarity_distance, Fc2MostSimilarity_index = torch.topk(SoftmaxFc2Distance_matrix,
                                                                         self.top_number,dim=1,largest=False)
        Fc2MostSimilarity_index = Fc2MostSimilarity_index[:, 1:]
        Fc2MostSimilarity_index = torch.flatten(Fc2MostSimilarity_index)
        Featurememory2 = self.target_featurememory[Fc2MostSimilarity_index,:]

        return Featurememory1,Featurememory2


    def predict(self, target_feature,fc1,fc2,index_target):
        fc1 = F.softmax(fc1)
        fc2 = F.softmax(fc2)

        '''        Update the Memory at first        '''
        self.update(target_feature, fc1, fc2, index_target)

        '''  L1 distance with fc1softmax and fc2softmax    '''
        Fc1memory, Fc2memory = self.feature_smiliarity(target_feature, index_target)
        loss_softmax = self.Discrepancy(Fc1memory, Fc2memory)#L1损失

        ''' feature distance with fc1feature and fc2feature '''
        Featurememory1, Featurememory2 = self.softmax_smiliarity(fc1,fc2, index_target)
        loss_feature = self.FeatureDistance(Featurememory1, Featurememory2)
        loss_feature = torch.mean(loss_feature)

        return loss_softmax, loss_feature


    def FeatureDistance(self, pointA, pointB):
        '''
        Calcuate the Normal Distance with centers and centers
        :param pointA: [*,feature_dim]
        :param pointB: [*,feature_dim]
        :return: []
        '''
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        Distance = 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        return Distance

    def CosDistance(self,pointA,pointB):
        '''
        Calcuate the Cosine Distance with features and (centers,features)
        :param pointA:[batchsize,feature_dim]
        :param pointB:[batchsize,feature_dim] or [class_num,feature_dim]
        :return:[batchsize,batchsize] or [batchsize,class_num]
        '''
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)

        assert (pointA.size(1) == pointB.size(1))
        cosdistance = 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
        return cosdistance

    def Discrepancy(self,softmax1,softmax2):
        '''L1 loss'''
        loss = torch.mean(torch.abs(softmax1-softmax2))
        return loss



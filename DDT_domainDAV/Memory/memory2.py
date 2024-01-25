import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class Memory(nn.Module):
    def __init__(self, class_number, feature_dim, target_number, source_number, top_number):
        super(Memory, self).__init__()
        self.class_number = class_number
        self.feature_dim = feature_dim
        self.target_number = target_number
        self.source_number = source_number
        self.top_number = top_number
        self.momentum = True

        if torch.cuda.is_available():
            '''center_featurememory:特征中心记忆模块(源域和目标域公用):[class_number,feature_dim]'''
            self.center_featurememory = torch.zeros(self.class_number, self.feature_dim).cuda()
            '''refs:使用refs更新特征中心'''
            self.refs = to_cuda(torch.LongTensor(range(self.class_number)).unsqueeze(1))

            '''target_featurememory:目标域数据的特征记忆模块:[target_number,feature_dim]'''
            self.target_featurememory = torch.zeros(self.target_number, self.feature_dim).cuda()
            '''target_classmemory:目标域数据的标签记忆模块:[target_number]'''
            self.target_classmemory = torch.zeros(self.target_number).cuda().long()

            '''source_featurememory:源域数据的特征记忆模块:[source_number,feature_dim]'''
            self.source_featurememory = torch.zeros(self.source_number, self.feature_dim).cuda()
            '''source_classmemory:源域数据的标签记忆模块:[target_number]'''
            self.source_classmemory = torch.zeros(self.source_number).cuda().long()

            '''source_class2center:源域数据中特征与中心的距离的记忆模块[class_number,0:个数,1:距离总和]'''
            self.source_class2center = torch.zeros(self.class_number,2).cuda()
            '''source_interdistance:源域数据中记录第一距离和第二距离之间的差值的记忆模块[class_number,0:个数,1:距离总和]'''
            self.source_interdistance = torch.zeros(self.class_number,self.class_number,2).cuda()

    def featuredistance2center(self,feature_source,label_source):
        '''首先计算源域特征和聚类中心的距离,使用source_class2center记录特征和特征中心的距离'''
        dists_centers_update = self.CosDistance(feature_source, self.center_featurememory)
        batch = int(feature_source.size()[0])
        for i in range(batch):
            label = label_source[i]
            distance = dists_centers_update[i, label]
            self.source_class2center[label,0] =  self.source_class2center[label,0] + 1
            self.source_class2center[label,1] =  self.source_class2center[label,1] + distance


    def update_center(self, feature, labels, labels_value):
        '''使用特征(源域特征:有真实标签)(目标域特征:预测的伪标签和预测概率)更新特征中心记忆'''
        feature = feature.data
        labels = labels.unsqueeze(0).expand(self.class_number, -1)
        mask = (labels == self.refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)

        if labels_value is None:
            feature = feature
        else:
            labels_value = labels_value.unsqueeze(0).reshape((-1, 1))
            feature = feature * labels_value

        self.center_featurememory +=  torch.sum(feature * mask, dim=1)

    def update_target(self, feature, index_target, label_target):
        '''使用目标域特征更新目标域特征记忆模块'''
        self.target_classmemory[index_target] = label_target.data
        if self.momentum:
            momentum = 0.1
            self.target_featurememory[index_target] = (1.0 - momentum) * self.target_featurememory[index_target] + momentum * feature.data
        else:
            self.target_featurememory[index_target, :] = feature.data

    def update_source(self, feature, index_source,label_source):
        '''使用源域特征更新源域特征记忆模块'''
        self.source_classmemory[index_source] = label_source.data
        if self.momentum:
            momentum = 0.1
            self.source_featurememory[index_source] = (1.0 - momentum) * self.source_featurememory[
                index_source] + momentum * feature.data
        else:
            self.source_featurememory[index_source, :] = feature.data

    def sourcefeature2differertcenter(self,source_feature,source_label):
        '''根据源域数据计算和特征中心的距离,计算距离第一个中心和第二个中心之间的距离差,记录两个中心和(特征和两个中心的差值)'''
        Dists_sourcefeature2center = self.CosDistance(source_feature,self.center_featurememory)
        feature2center_distances, feature2center_labels = torch.topk(Dists_sourcefeature2center,5, dim=1,largest=False)

        batch = int(source_feature.size()[0])
        for i in range(batch):
            top1 = feature2center_labels[i,0]
            top2 = feature2center_labels[i,1]
            top1_value = feature2center_distances[i, 0]
            top2_value = feature2center_distances[i, 1]

            Inter_distance = torch.abs(top1_value- top2_value)
            self.source_interdistance[top1, top2, 0] = self.source_interdistance[top1, top2, 0] + 1
            self.source_interdistance[top1, top2, 1] = self.source_interdistance[top1, top2, 1] + Inter_distance

    def feature2center(self, feature, labels):
        dists_centers_update = self.CosDistance(feature, self.center_featurememory)
        batch,number_class = dists_centers_update.size()
        loss = 0
        for i in range(batch):
            label = labels[i]
            loss += dists_centers_update[i,label]
        loss = loss/batch
        return loss

    def InterMask(self,center_labels1,center_labels2):
        '''根据sourcefeature2differertcenter和source_interdistance判断目标特征和两个特征中心的差异阈值Interhredhold'''
        batchsize = center_labels1.size()[0]
        thredholds = torch.zeros(batchsize).cuda()
        for i in range(batchsize):
            top1 = center_labels1[i]
            top2 = center_labels2[i]
            Distance_number = self.source_interdistance[top1, top2, 0]
            Distance_value = self.source_interdistance[top1,top2, 1]
            if Distance_number == 0:
                thredholds[i] = torch.tensor(0.05).cuda()
            else:
                thredhold = Distance_value/ Distance_number
                thredholds[i] = thredhold

        return thredholds

    def MaskAcc(self, mask, TrueLable, PredLabel):
        '''确定挑选的伪标签的准确度和挑选的数量'''
        #Selcted = int(torch.where(mask == True)[0].size()[0])
        #Selcted_Correct = int(torch.where((Selcted_True == Selcted_Pred)== True)[0].size()[0])

        Selcted = int((mask == True).nonzero().reshape((-1)).size()[0])
        Selcted_True = torch.masked_select(TrueLable,mask)
        Selcted_Pred = torch.masked_select(PredLabel,mask)
        Selcted_Correct = int(((Selcted_True == Selcted_Pred)== True).nonzero().reshape((-1)).size()[0])

        return Selcted,Selcted_Correct

    def feature_smiliarity(self, image_target, index_target,label_target,class_model):
        '''主程序'''

        '''将目标特征进行分类，确定标签的概率'''
        with torch.no_grad():
            output_target, feature_target = class_model(image_target)

        output_target = F.softmax(output_target,dim = 1)
        softmax_value, softmax_label = torch.topk(output_target,1,largest = True)
        softmax_value = softmax_value.flatten()

        '''计算目标数据和中心的距离'''
        dists_centers = self.CosDistance(feature_target, self.center_featurememory)
        center_distance,center_labels = torch.topk(dists_centers,5, dim=1,largest=False)

        '''计算每个类的距离,Center_mask'''
        class_number = self.source_class2center[center_labels[:,0]][:,0]
        class_value = self.source_class2center[center_labels[:,0]][:,1]
        Class_Distance = class_value/class_number

        Center_value_top1 = center_distance[:, 0]
        Center_mask = Center_value_top1 < Class_Distance
        Center_sum,Center_Corr = self.MaskAcc(Center_mask,label_target,center_labels[:,0])

        '''计算top1和top2之间的差距,计算Inter_mask'''
        Inter_distance = torch.abs(center_distance[:, 0] - center_distance[:, 1])
        Inter_mask = (Inter_distance > 0.05)
        Inter_sum,Inter_Corr = self.MaskAcc(Inter_mask, label_target, center_labels[:, 0])

        Inter_distance_thredhold = self.InterMask(center_labels[:, 0], center_labels[:, 1])
        Inter_thredholdmask = (Inter_distance > Inter_distance_thredhold)
        InterThre_sum, InterThre_Corr = self.MaskAcc(Inter_thredholdmask, label_target, center_labels[:, 0])

        '''合并Center_mask和计算Inter_mask'''
        fileter_mask = Center_mask & Inter_thredholdmask
        fileter_sum,fileter_Corr = self.MaskAcc(fileter_mask, label_target, center_labels[:, 0])

        '''寻找出最保险的目标域数据和标签'''
        pseudo_centerlabel = center_labels[:,0]

        MaskPseudo_feature = feature_target[fileter_mask,:]
        MaskPseudo_label  = torch.masked_select(pseudo_centerlabel,fileter_mask)
        MaskPseudo_value = torch.masked_select(softmax_value,fileter_mask)

        print('label_target:{}'.format(label_target))
        print('pseudo_label:{}'.format(pseudo_centerlabel))
        print('{}/{}'.format(fileter_Corr,fileter_sum))

        '''根据标签更新目标域的中心'''
        self.update_center(MaskPseudo_feature, MaskPseudo_label,MaskPseudo_value)

        ''''最小化目标域特征和特征中心的距离'''
        dists_centers_update = self.CosDistance(feature_target, self.center_featurememory)
        center_distance_update, center_labels_update = torch.min(dists_centers_update, dim=1)
        loss = torch.mean(center_distance_update)

        return loss

    def CosDistance(self, pointA, pointB):
        '''计算pointA和pointB之间的余弦距离'''
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)

        assert (pointA.size(1) == pointB.size(1))
        cosdistance = 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
        return cosdistance
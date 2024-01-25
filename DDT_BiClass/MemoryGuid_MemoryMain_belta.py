import os,sys
import torch
import argparse
import warnings

path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(path)

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

'''Model and dataloader'''
from Model.model import Generator,Classifier
from Model.data_loader import load_data_ImageFolder,load_data_ImageFolder_Index
from Model.PairedData import CVDataLoader
from Model.Memory import Memory
from Model.Loss import CrossEntropyLabelSmooth

'''Distance and Entropy'''
from Utils.Distance import BCDM_Discrepancy, MCD_Discrepancy
from Utils.Entropy import MME_entropy,Entropy

parser = argparse.ArgumentParser(description='Memory Guild DA')
parser.add_argument('--batch-size',type=int,default = 32)
parser.add_argument('--method',type=str, default = 'BCDM', help='BCDM or MCD')
parser.add_argument('--day',type=str, default = '0613')
parser.add_argument('--time',type=str, default = '0943')

parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--max_epoch',type=int,default=40)
parser.add_argument('--optimizer',type=str,default='SGD')
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--weight_decay',type=float,default=0.0005)

parser.add_argument('--cuda',default=True)
parser.add_argument('--num_k',type=int,default=4)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--dataset',type=str,default='office31')
parser.add_argument('--source',type=str)
parser.add_argument('--target',type=str)
parser.add_argument('--root_path', type=str)
parser.add_argument('--num_class', type=int)
parser.add_argument('--topK', type=int,default=5)
parser.add_argument('--initepoch', default=True)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

warnings.filterwarnings('ignore')

def update_optimizer(optimizer_g,optimizer_f,optimizer_dype,num_epoches,max_epoch,
                    lr, gamma, pow = 0.75 , weight_decay = 0.0005):

    if optimizer_dype == 'SGD':
        if num_epoches != 999999:
            p = (num_epoches-1)/max_epoch

        lr = lr * (1.0 + gamma*p)**(-pow)
    else:
        lr = lr

    cur_lr = lr
    for param_group in optimizer_g.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    for param_group in optimizer_f.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    return cur_lr

def train_epoch(args, epoch_idx, train_dataset, model, optimizer, MemoryGuid):
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = CrossEntropyLabelSmooth(reduction='none',num_classes=args.num_class, epsilon=0.1)
    Genera_model, class1_model, class2_model = model

    optimizer_g,optimizer_f = optimizer
    cur_lr = update_optimizer(optimizer_g,optimizer_f, args.optimizer, epoch_idx, args.max_epoch, lr=args.lr, gamma=10)

    Genera_model.train()
    class1_model.train()
    class2_model.train()

    for batch_idx, data in enumerate(train_dataset):

        if args.cuda:
            img_source = data['S'].cuda()
            index_source = data['S_index'].cuda()
            label_source = data['S_label'].cuda()

            img_target = data['T'].cuda()
            index_target = data['T_index'].cuda()
            label_target = data['T_label'].cuda()

        '''First step'''
        #source image
        feature_source = Genera_model(img_source)
        output_s1 = class1_model(feature_source)
        output_s2 = class2_model(feature_source)

        loss_s1 = criterion(output_s1, label_source)
        loss_s2 = criterion(output_s2, label_source)
        loss = loss_s1 + loss_s2

        #target image
        feature_target = Genera_model(img_target)
        output_t1 = class1_model(feature_target)
        output_t2 = class2_model(feature_target)
        entropy_loss = Entropy(output_t1,output_t2)

        #loss
        First_loss = loss + 0.01 * entropy_loss

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        First_loss.backward()
        optimizer_g.step()
        optimizer_f.step()


        '''Second step'''
        #source Image
        with torch.no_grad():
            feature_source = Genera_model(img_source)

        output_s1 = class1_model(feature_source)
        output_s2 = class2_model(feature_source)

        loss_s1 = criterion(output_s1, label_source)
        loss_s2 = criterion(output_s2, label_source)
        loss = loss_s1 + loss_s2

        #Target Image
        with torch.no_grad():
            feature_target = Genera_model(img_target)

        output_t1 = class1_model(feature_target)
        output_t2 = class2_model(feature_target)

        entropy_loss = Entropy(output_t1,output_t2)
        
        #MemoryLoss
        if args.initepoch:
            if epoch_idx == 1:
                softmax_loss, feature_loss = MemoryGuid.predict(feature_target, output_t1, output_t2, index_target)
                softmax_loss = softmax_loss * 0
            else:
                softmax_loss, feature_loss = MemoryGuid.predict(feature_target,output_t1,output_t2,index_target)
                softmax_loss = 1.0 * softmax_loss 
        else:
            softmax_loss, feature_loss = MemoryGuid.predict(feature_target,output_t1,output_t2,index_target)
            softmax_loss = 1.0 * softmax_loss 


        #Discrepancy
        if args.method == 'BCDM':
            dis_loss = 0.01 * BCDM_Discrepancy(output_t1, output_t2)
        elif args.method == 'MCD':
            dis_loss = MCD_Discrepancy(output_t1, output_t2)

        Second_loss = loss - (dis_loss + softmax_loss) + 0.01 * entropy_loss
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        Second_loss.backward()
        optimizer_f.step()


        '''Three Step'''
        for i in range(args.num_k):

            #Target Image
            feature_target = Genera_model(img_target)
            output_t1 = class1_model(feature_target)
            output_t2 = class2_model(feature_target)

            entropy_loss = 0.01 * Entropy(output_t1, output_t2)

            #Discrepancy
            if args.method == 'BCDM':
                dis_loss = 0.01 * BCDM_Discrepancy(output_t1, output_t2)
            elif args.method == 'MCD':
                dis_loss = MCD_Discrepancy(output_t1, output_t2)

            #MemoryLoss
            if args.initepoch:
                if epoch_idx == 1:
                    softmax_loss, feature_loss = MemoryGuid.predict(feature_target, output_t1, output_t2, index_target)
                    feature_loss = feature_loss * 0
                else:
                    softmax_loss, feature_loss = MemoryGuid.predict(feature_target,output_t1,output_t2,index_target)
                    feature_loss = 1.0 * feature_loss
            else:
                softmax_loss, feature_loss = MemoryGuid.predict(feature_target,output_t1,output_t2,index_target)
                feature_loss = 1.0 * feature_loss

            
            '''Total Loss'''
            Third_loss = dis_loss +  entropy_loss + feature_loss

            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            Third_loss.backward()
            optimizer_g.step()


        if batch_idx % 10 == 0:
            print('Epoch:{}/{},Iter:{},lr:{},'
                  'Loss1:{:.4f},Loss2:{:.4f},'
                  'Discrepancy:{:.4f},Entropy_loss:{:.4f},SoftmaxMemory_loss:{:.4f},FeatureMemory_loss:{:.4f}'.format(
                epoch_idx, args.max_epoch,batch_idx,cur_lr,
                loss_s1.item(),loss_s2.item(),
                dis_loss.item(),entropy_loss.item(),
                softmax_loss.item(),feature_loss.item()))

def test_epoch(epoch_idx,model,test_loader,SourceOrTarget,save_dir):
    Genera_model, class1_model, class2_model = model

    Genera_model.eval()
    class1_model.eval()
    class2_model.eval()

    correct1 = 0
    correct2 = 0
    correct3 = 0
    test_loss = 0
    size = 0

    for batch_idx, (test_data, test_label, test_index) in enumerate(test_loader):
        data, target = test_data.cuda(), test_label.cuda()

        data_feature = Genera_model(data)
        output_fc1 = class1_model(data_feature)
        output_fc2 = class2_model(data_feature)

        '''Class 1'''
        test_loss += F.nll_loss(F.log_softmax(output_fc1, dim=1), target, size_average=False).item()  # sum up batch loss
        pred1 = output_fc1.data.max(1)[1]

        correct1 += pred1.eq(target.data).cpu().sum()

        '''Class 2'''
        pred2 = output_fc2.data.max(1)[1]
        correct2 += pred2.eq(target.data).cpu().sum()

        '''Class ensemble'''
        output_ensemble = output_fc1 + output_fc2
        pred_ensemble = output_ensemble.data.max(1)[1]
        correct3 += pred_ensemble.eq(target.data).cpu().sum()

        k = target.data.size()[0]
        size += k

    test_loss = test_loss / size
    acc_c1 = float(100. * correct1.item() / size)
    acc_c2 = float(100. * correct2.item() / size)
    acc_ensemble = float(100. * correct3.item() / size)

    print(
        'Epoch:{}, Test set:{}, Average loss:{:.4f},Accuracy C1:{}/{}({:.2f}%),Accuracy C2:{}/{}({:.2f}%),Accuracy Ensemble:{}/{}({:.2f}%)'.format(
        epoch_idx,SourceOrTarget,            test_loss, correct1, size, acc_c1,       correct2, size, acc_c2,         correct3, size, acc_ensemble
        ))
    save_txt = 'Epoch:{}_TestSet:{}_AccuracyC1:{:.2f}%_AccuracyC2:{:.2f}%_AccuracyEnsemble:{:.2f}%.txt'.format(
        epoch_idx,SourceOrTarget, acc_c1, acc_c2, acc_ensemble
    )
    save_path = os.path.join(save_dir,save_txt)
    os.mknod(save_path)
    return acc_c1,acc_c2,acc_ensemble

def save_best(save_dir,G,F1,F2):
    g_path = os.path.join(save_dir,'G.pth')
    f1_path = os.path.join(save_dir,'F1.pth')
    f2_path = os.path.join(save_dir,'F2.pth')

    if os.path.exists(g_path):
        os.remove(g_path)
    if os.path.exists(f1_path):
        os.remove(f1_path)
    if os.path.exists(f2_path):
        os.remove(f2_path)

    torch.save(G.state_dict(), g_path)
    torch.save(F1.state_dict(), f1_path)
    torch.save(F2.state_dict(), f2_path)

if __name__ == '__main__':
    if args.dataset == 'office31':
        args.root_path = '/raid/huangl02/WGQ/DA_data/office31/'
    if args.dataset == 'officehome':
        args.root_path = '/raid/huangl02/WGQ/DA_data/OfficeHome/'
    if args.dataset == 'CLEF':
        args.root_path = '/raid/huangl02/WGQ/DA_data/Image_CLEF/'
    if args.dataset == 'visda':
        args.root_path = '/raid/huangl02/WGQ/DA_data/visda17_pick'
    print(args)
    print('WGQ MemoryDA: BestResult:Source:{}-->Target:{}'.format(args.source, args.target))
    save_dir = './MemoryResultBelta_VisDA/Memory_{}_{}_{}_{}/Classification_{}_{}'.format(
        args.dataset, args.method, 
        args.day,args.time,
        args.source, args.target
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(os.path.join(save_dir,'args.txt'), 'wt') as f:
        print(args, file=f)

    '''Init Dataset'''
    train_loader, test_loader, dataset = load_data_ImageFolder_Index(args, kwargs)
    num_source = len(dataset[0].imgs)
    num_target = len(dataset[1].imgs)
    print('Source:{},len:{}'.format(args.source, num_source))
    print('Target:{},len:{}'.format(args.target, num_target))

    source_train_loader, target_train_loader = train_loader
    source_test_loader, target_test_loader = test_loader
    dataset_S, dataset_T = dataset

    train_loader = CVDataLoader()
    train_loader.initialize(source_train_loader, target_train_loader, dataset_S, dataset_T)
    train_dataset = train_loader.load_data()
    args.num_class = len(source_train_loader.dataset.classes)

    '''Init Model'''
    Genera = Generator(base_net='ResNet101').cuda()
    class1 = Classifier(number_classes=args.num_class).cuda()
    class2 = Classifier(number_classes=args.num_class).cuda()
    model = [Genera, class1, class2]

    '''Init optimizer'''
    optimizer_g = optim.SGD(params=Genera.get_parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True )
    optimizer_f = optim.SGD(params=list(class1.get_parameters()) + list(class2.get_parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = [optimizer_g, optimizer_f]
    acc_best = 0.

    '''Init Memory'''
    MemoryGuid = Memory(class_number=args.num_class, feature_dim = 256, target_number = num_target, top_number = args.topK )
    MemoryGuid.init_memory(Genera, target_test_loader)

    '''Start Train'''
    for epoch_idx in range(1, args.max_epoch + 1):
        train_epoch(args, epoch_idx, train_dataset, model, optimizer,MemoryGuid)
        
        if epoch_idx % 1 == 0:
            test_epoch(epoch_idx, model, source_test_loader, 'Source', save_dir)
            acc_c1, acc_c2, acc_ensemble = test_epoch(epoch_idx, model, target_test_loader, 'Target', save_dir)
            acc_best_epoch = max(acc_c1, acc_c2, acc_ensemble)
            if acc_best_epoch > acc_best:
                acc_best = acc_best_epoch
                save_best(save_dir, Genera, class1, class2)
            print('Best:{:.8f}'.format(acc_best))
            print('-' * 130)
    Best_save_txt = 'Source_{}_Target_{}_BestAcc_{:.2f}%.txt'.format(
        args.source, args.target, acc_best
    )
    Best_save_path = os.path.join(save_dir, Best_save_txt)
    os.mknod(Best_save_path)

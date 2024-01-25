import random
import time
import warnings
import sys
import argparse
import copy
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')
'''Memory'''
from Memory.memory1 import Memory

'''DANN'''
import models as models

from utils.domain_discriminator import DomainDiscriminator
from model.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.lr_scheduler import StepwiseLR

from Data.data_loader import load_data_ImageFolder_Index
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Memory Guild DA')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--method', type=str, default='CDAN_memory', help='CDAN or CDAN_memory')
parser.add_argument('--day', type=str, default='0613')
parser.add_argument('--time', type=str, default='0943')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--max_epoch', type=int, default=40)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--cuda', default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='office31')
parser.add_argument('--source', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--root_path', type=str)
parser.add_argument('--num_class', type=int)
parser.add_argument('--topK', type=int, default=5)
parser.add_argument('--iters-per-epoch', default=1000, type=int)
parser.add_argument('--trade-off', default=1., type=float)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
print(args)
warnings.filterwarnings('ignore')


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: ConditionalDomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace, Memory):

    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        image_s, labels_s, index_s = next(train_source_iter)
        image_t, labels_t, index_t = next(train_target_iter)

        image_s = image_s.to(device)
        image_t = image_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # compute output
        image_st = torch.cat((image_s, image_t), dim=0)
        fc_st, feature_st = model(image_st)

        fc_s, fc_t = fc_st.chunk(2, dim=0)
        feature_s, feature_t = feature_st.chunk(2, dim=0)

        '''记忆模块,输出记忆损失'''
        if epoch == 0:
            Memory.update_center(feature_s, labels_s, None)
            Memory.featuredistance2center(feature_s, labels_s)
            Memory.sourcefeature2differertcenter(feature_s, labels_s)
            memoryloss = torch.tensor(0).cuda()
        else:
            Memory.update_center(feature_s, labels_s, None)
            Memory.featuredistance2center(feature_s, labels_s)
            Memory.sourcefeature2differertcenter(feature_s, labels_s)
            memoryloss = Memory.feature_smiliarity(image_t, index_t, labels_t, classifier)

        cls_loss = F.cross_entropy(fc_s, labels_s)

        transfer_loss = domain_adv(fc_s, feature_s, fc_t,feature_t)
        domain_acc = domain_adv.domain_discriminator_accuracy

        loss = cls_loss + transfer_loss * args.trade_off + memoryloss

        cls_acc = accuracy(fc_s, labels_s)[0]

        losses.update(loss.item(), image_s.size(0))
        cls_accs.update(cls_acc.item(), image_s.size(0))
        domain_accs.update(domain_acc.item(), image_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch[{}/{}][{}/{}],Cls_loss:{:.4f}, Transfer_loss:{:.4f},Memoryloss:{:.4f}'.format(
                epoch, args.max_epoch,
                i, args.iters_per_epoch,
                cls_loss.item(), transfer_loss.item(), memoryloss.item()
            ))


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace,
             epoch: int, save_dir: str, writer) -> float:
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        for i, (images, target, index) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % 100 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    writer.add_scalar('Loss/Validation', losses.avg, epoch)
    writer.add_scalar('Accuracy_class/Validation', top1.avg, epoch)

    save_txt = 'Epoch:{}_AccuracyTop1:{:.2f}%_AccuracyTop5:{:.2f}%.txt'.format(
        epoch, top1.avg, top5.avg)
    save_path = os.path.join(save_dir, save_txt)
    os.mknod(save_path)

    return top1.avg


def delate(root_dir):
    AccList = []
    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.pth':
                name = os.path.splitext(filename)[0]
                acc = float(name.split('_')[2])
                AccList.append(acc)

    max_acc = max(AccList)

    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.pth':
                name = os.path.splitext(filename)[0]
                acc = float(name.split('_')[2])
                if acc != max_acc:
                    os.remove(os.path.join(root_dir, filename))


def initmemory(source_test_loader, classifier, MemoryGuid):
    classifier.eval()

    for batch_idx, (test_data, test_label, test_index) in enumerate(source_test_loader):
        test_data, test_label = test_data.cuda(), test_label.cuda()
        source_fc, source_feature = classifier(test_data)
        MemoryGuid.update_center(source_feature, test_label, None)


if __name__ == '__main__':

    if args.dataset == 'office31':
        args.root_path = '/raid/huangl02/WGQ/DA_data/office31/'
    if args.dataset == 'officehome':
        args.root_path = '/raid/huangl02/WGQ/DA_data/OfficeHome/'
    if args.dataset == 'CLEF':
        args.root_path = '/raid/huangl02/WGQ/DA_data/Image_CLEF/'
    if args.dataset == 'VisDA':
        args.root_path = '/raid/huangl02/WGQ/DA_data/visda17/'

    print('WGQ MemoryDA: BestResult:Source:{}-->Target:{}'.format(args.source, args.target))
    save_dir = './Memory1_banchmark/CDAN_memory1/Memory_{}_{}_{}_{}/save/Classification_{}_{}'.format(
        args.dataset, args.method,
        args.day, args.time,
        args.source, args.target
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_dir = './Memory1_banchmark/CDAN_memory1/Memory_{}_{}_{}_{}/log/Classification_{}_{}'.format(
        args.dataset, args.method,
        args.day, args.time,
        args.source, args.target
    )

    # SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(save_dir, 'args.txt'), 'wt') as f:
        print(args, file=f)

    '''Init Dataset'''
    # Data loading code
    train_loader, test_loader, dataset = load_data_ImageFolder_Index(args, kwargs)
    num_source = len(dataset[0].imgs)
    num_target = len(dataset[1].imgs)

    source_train_loader, target_train_loader = train_loader
    source_test_loader, target_test_loader = test_loader
    dataset_S, dataset_T = dataset
    args.num_class = len(source_train_loader.dataset.classes)

    train_source_iter = ForeverDataIterator(source_train_loader)
    train_target_iter = ForeverDataIterator(target_train_loader)

    print('Source:{},len:{},iter:{}'.format(args.source, num_source, train_source_iter.__len__()))
    print('Target:{},len:{},iter:{}'.format(args.target, num_target, train_target_iter.__len__()))

    '''Init Model'''
    if args.dataset == 'VisDA':
        backbone = models.__dict__['resnet101'](pretrained=True)
    else:
        backbone = models.__dict__['resnet50'](pretrained=True)

    classifier = ImageClassifier(backbone, args.num_class).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim * args.num_class, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(domain_discri,
                                                  entropy_conditioning=False,
                                                  num_classes=args.num_class,
                                                  features_dim=classifier.features_dim,
                                                  randomized=False).to(device)

    '''初始化记忆模块'''
    MemoryGuid = Memory(class_number=args.num_class, feature_dim=256, target_number=num_target,
                        source_number=num_source, top_number=args.topK)
    initmemory(source_test_loader, classifier, MemoryGuid)

    # start training
    best_acc1 = 0.
    for epoch in range(args.max_epoch):
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args, MemoryGuid)
        acc1 = validate(target_test_loader, classifier, args, epoch, save_dir, writer)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
            torch.save({'epoch': epoch + 1,
                        'state_dict': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': acc1},
                       os.path.join(save_dir, 'model_{}_{:.3f}.pth'.format(epoch + 1, acc1))
                       )

        best_acc1 = max(acc1, best_acc1)

    delate(save_dir)
    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(target_test_loader, classifier, args, epoch + 1, save_dir, writer)
    print("test_acc1 = {:3.1f}".format(acc1))
    writer.close()

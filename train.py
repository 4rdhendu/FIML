# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from statistics import mean
# import multiprocessing

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.fusion import fuse_prob, fuse_score
from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12
from models.WideResNet28_10_embedding import WideResNet
from models.semantic_classifier import SemanticClassifier
from tensorboardX import SummaryWriter

from models.dimp_heads import dimp_norm_init_shannon_hingeL2Loss

from utils import set_gpu, Timer, count_accuracy, check_dir, log

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(a.device))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        # 256 dimensional
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        # 8192 dimensional
        network = R2D2Embedding().cuda()
    elif options.network == 'WideResNet_DC':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            if options.aws == 1:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=21).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=21).cuda()
                network = torch.nn.DataParallel(network)
        else:
            if options.aws == 1:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=8).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=8).cuda()
                network = torch.nn.DataParallel(network)
    elif options.network == 'ResNet_DC':
        # 2560 dimensional
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            if options.aws == 1:
                network = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, avgpool_param=5).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, avgpool_param=5).cuda()
                network = torch.nn.DataParallel(network)
        else:
            if options.aws == 1:
                network = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, avgpool_param=2).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, avgpool_param=2).cuda()
                network = torch.nn.DataParallel(network)
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'FIML':
        cls_head = dimp_norm_init_shannon_hingeL2Loss(num_iter=options.steepest_descent_iter, norm_feat=options.norm_feat,
                                                      entropy_weight=options.entropy_weight, entropy_temp=options.entropy_temp,
                                                      learn_entropy_weights=options.learn_entropy_weights,
                                                      learn_entropy_temp=options.learn_entropy_temp, learn_weights=options.learn_weights,
                                                      pos_weight=options.pos_weight, neg_weight=options.neg_weight,
                                                      learn_slope=options.learn_slope,
                                                      pos_lrelu_slope=options.pos_lrelu_slope,
                                                      neg_lrelu_slope=options.neg_lrelu_slope,
                                                      learn_spatial_weight=options.learn_inner_spatial_weight,
                                                      dc_factor=options.dc_factor)
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':

    # multiprocessing.set_start_method('spawn', force=True)
    # cv.setNumThreads(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')

    # val episodes bhi ek param hai
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--tensorboard-dir', default='./experiments/exp_1/tensorboard')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ResNet_DC',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')

    parser.add_argument('--head', type=str, default='FIML',
                            help='choose which classification head to use')
    parser.add_argument('--train-loss', type=str, default='CrossEntropy',
                        help='choose which loss to use. SmoothedCrossEntropy, CrossEntropy')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=16,
                            help='number of episodes per batch')

    # the eps parameter is for label smoothing
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--steepest-descent-iter', type=int, default=5, help='number of steepest descent iterations')
    parser.add_argument('--load-path', type=str, default='./experiments/exp_1', help='load path')



    parser.add_argument('--norm-feat', type=int, default=1,
                        help='to normalize the features or not. 1, 0')

    parser.add_argument('--transductive-learning', type=str, default='True',
                        help='Employ transductive learning or not. True, False')

    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer ADAM/SGD')

    parser.add_argument('--learn-rate', type=float, default=0.1, help='starting learning rate for the backbone ()')
    parser.add_argument('--debug', type=str, default='True',
                        help='Whether we are debugging or not. False, True')

    # This is going to be used for the different way of calling the the heads encapsulation
    parser.add_argument('--algorithm-type', type=str, default='DiMP',
                        help='Are we using the DiMP class of algos or other: DiMP, Other')
    parser.add_argument('--smooth-label-base-learner', type=str, default='False', help='whether to use the smooth labels inside the base learner or not')
    parser.add_argument('--smooth-label-factor-base-learner', type=float, default=0.0, help='smooth factor for label smoothing in base learner')

    parser.add_argument('--aws', type=int, default=0,
                        help='whether we are using aws or not')

    parser.add_argument('--entropy-weight', type=float, default=1e3, help='entropy weight')
    parser.add_argument('--entropy-temp', type=float, default=1.0, help='entropy temp')
    parser.add_argument('--learn-entropy-weights', type=str2bool, default=True, help='whether to learn entropy weights')
    parser.add_argument('--learn-entropy-temp', type=str2bool, default=True, help='whether to learn entropy temp')

    parser.add_argument('--pos-weight', type=float, default=1, help='pos example weight')
    parser.add_argument('--neg-weight', type=float, default=1, help='neg example weight')
    parser.add_argument('--pos-lrelu-slope', type=float, default=1.0, help='positive relu slope')
    parser.add_argument('--neg-lrelu-slope', type=float, default=1.0, help='negative relu slope')
    parser.add_argument('--learn-weights', type=str2bool, default=True, help='whether to learn pos neg weights')
    parser.add_argument('--learn-slope', type=str2bool, default=True, help='whether to learn relu slope')

    # dense classification
    parser.add_argument('--DC-support', type=str2bool, default=True,
                        help='whether to use dense classification at support level')

    parser.add_argument('--DC-query-inner', type=str2bool, default=True,
                        help='whether to use dense classification at query level')

    parser.add_argument('--train-query-loss', type=int, default=1,
                        help='0 for true query samples and for dense loss, '
                             '1 for the dense classification with fusion of score, '
                             '2 with fusion of probability')

    parser.add_argument('--val-fusion-module', type=int, default=1,
                        help='0 is for the true query samples, 1 is for score fusion, 2 is for probability fusion')

    parser.add_argument('--learn-fusion-weight', type=str2bool, default=True,
                        help='whether to learn the fusion weights or not')

    parser.add_argument('--learn-inner-spatial-weight', type=str2bool, default=True,
                        help='whether to learn the fusion weights or not')

    parser.add_argument('--stop-class-iter', type=int, default=0, help='number of iterations for luke warm training')

    opt = parser.parse_args()

    # print("Device Count: ", torch.cuda.device_count())

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch

    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    if opt.aws == 1:
        set_gpu(opt.gpu)

    # check_dir('./experiments/')
    check_dir(opt.save_path)
    check_dir(opt.tensorboard_dir)

    # debug the GPU part
    # print("Device Count: ", torch.cuda.device_count())

    # print("Dev 1: ", torch.cuda.get_device_name(0))
    # print("Dev 2: ", torch.cuda.get_device_name(1))
    # print("Dev 3: ", torch.cuda.get_device_name(2))
    # print("Dev 4: ", torch.cuda.get_device_name(3))

    log_file_path = os.path.join(opt.save_path, "train_log.txt")

    print(log_file_path)

    log(log_file_path, str(vars(opt)))

    dc_factor = 1
    if opt.network == 'ResNet_DC':
        if opt.dataset == 'miniImageNet' or opt.dataset == 'tieredImageNet':
            dc_factor = 25
        else:
            dc_factor = 4
    elif opt.network == 'WideResNet_DC':
        if opt.dataset == 'miniImageNet' or opt.dataset == 'tieredImageNet':
            dc_factor = 441
        else:
            dc_factor = 64

    opt.dc_factor = dc_factor
    (embedding_net, cls_head) = get_model(opt)

    if opt.train_query_loss == 1 or opt.val_fusion_module == 1:
        fusion_mod = fuse_score(dc_factor=dc_factor, weight_learn=opt.learn_fusion_weight).cuda()
    elif opt.train_query_loss == 2 or opt.val_fusion_module == 2:
        fusion_mod = fuse_prob(dc_factor=dc_factor, weight_learn=opt.learn_fusion_weight).cuda()
    else:
        fusion_mod = fuse_score(dc_factor=dc_factor, weight_learn=opt.learn_fusion_weight).cuda()


    if opt.optimizer=='SGD':
        optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                         {'params': cls_head.parameters()},
                                         {'params': fusion_mod.parameters()}], lr=opt.learn_rate, momentum=0.9, \
                                        weight_decay=5e-4, nesterov=True)

    else:
        optimizer = torch.optim.Adam([{'params': embedding_net.parameters()},
                                          {'params': cls_head.parameters()},
                                          {'params': fusion_mod.parameters()}], lr=opt.learn_rate)

    lambda_epoch = lambda e: 1.0 if e < 20 + opt.stop_class_iter else (0.06 if e < 40 + opt.stop_class_iter else 0.012 if e < 50 + opt.stop_class_iter else (0.0024))
    # lambda_epoch = lambda e: 1.0 if e < 10 + opt.stop_class_iter else (0.3 if e < 20 + opt.stop_class_iter
    #                                                                    else 0.1 if e < 30 + opt.stop_class_iter
    #                                                                    else 0.06 if e < 40 + opt.stop_class_iter
    #                                                                    else 0.012 if e < 50 + opt.stop_class_iter
    #                                                                    else (0.0024))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # tensorboard writer
    writer = SummaryWriter(log_dir=opt.tensorboard_dir)

    # This is where things need to change in the data loader
    train_loader = dloader_train.get_dataloader()
    val_loader = dloader_val.get_dataloader()

    # if opt.luke_warm == 'True' and opt.classification == 'True':
        # num_epoch = opt.num_epoch + opt.stop_class_iter
    # else:
    num_epoch = opt.num_epoch

    # loss * opt.few_shot_loss_weight + loss_class * opt.classifier_loss_weight

    fs_loss_wt = opt.few_shot_loss_weight
    cls_loss_wt = opt.classifier_loss_weight

    for epoch in range(1, num_epoch + 1):

        # Train on the training split
        lr_scheduler.step()


        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        # for i, batch in enumerate(dloader_train(epoch), 1):
        # for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
        for i, batch in enumerate(train_loader):
            data_support, labels_support, data_query, labels_query, Kall, nBase, Kabs = [x.cuda() for x in batch]

            if opt.DC_support:
                train_n_support = opt.train_way * opt.train_shot * dc_factor
            else:
                train_n_support = opt.train_way * opt.train_shot

            if opt.DC_query_inner:
                train_n_query = opt.train_way * opt.train_query * dc_factor
            else:
                train_n_query = opt.train_way * opt.train_query

            if not opt.DC_support:
                emb_support, _ = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            else:
                _, emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                labels_support = tile(labels_support, 1, dc_factor)

            # class_score = classifier(emb_support)


            if not opt.DC_query_inner:
                emb_query, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                query_num_samples = emb_query.shape[1]
            else:
                _, emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                labels_query = tile(labels_query, 1, dc_factor)

            # logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)
            if opt.algorithm_type == 'DiMP':
                # logit_query, inner_losses = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot *
                #                            (opt.dropout_samples_per_class_train+1+offset_shot_support))
                logit_query, inner_losses = cls_head(emb_query, emb_support, labels_support, opt.train_way,
                                                     int(emb_support.shape[1]/opt.train_way))
            else:
                # logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way,
                #                                      opt.train_shot*(opt.dropout_samples_per_class_train+1))
                logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way,
                                       int(emb_support.shape[1]/opt.train_way))
            if opt.train_loss == 'SmoothedCrossEntropy':

                if opt.train_query_loss == 1 or opt.train_query_loss == 2:
                    #labels_query = labels_query.view(labels_query.shape[0], labels_query.shape[1]/dc_factor, -1)
                    labels_query = labels_query[:, range(0, labels_query.shape[1], dc_factor)]
                    logit_query[-1] = fusion_mod(logit_query[-1])

                smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

                if opt.algorithm_type == 'Other':
                    log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
                else:
                    log_prb = F.log_softmax(logit_query[-1].reshape(-1, opt.train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
            else:
                if opt.algorithm_type == 'Other':
                    loss = x_entropy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
                else:
                    loss = x_entropy(logit_query[-1].reshape(-1, opt.train_way), labels_query.reshape(-1))


            if opt.algorithm_type == 'Other':
                acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
            else:
                #labels_query = labels_query[:, :query_num_samples]
                #logit_query[-1] = logit_query[-1][:, :query_num_samples]

                acc = count_accuracy(logit_query[-1].reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if i % 100 == 0:
                train_acc_avg = np.mean(np.array(train_accuracies))
                print(log_file_path)
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))


                # if opt.debug == 'True':
                #     print("The sgd losses:")
                #     for i_iter in range(len(inner_losses)):
                #         print("Iter " + str(i_iter) + ": " + str(inner_losses[i_iter]))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_avg = loss.item()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []

        #
        with torch.autograd.no_grad():
            for i, batch in enumerate(val_loader):
            # for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _, _ = [x.cuda() for x in batch]

                if opt.DC_support:
                    test_n_support = opt.test_way * opt.val_shot * dc_factor
                else:
                    test_n_support = opt.test_way * opt.val_shot

                if opt.DC_query_inner:
                    test_n_query = opt.test_way * opt.val_query * dc_factor
                else:
                    test_n_query = opt.test_way * opt.val_query

                if not opt.DC_support:
                    emb_support, _ = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                    emb_support = emb_support.reshape(1, test_n_support, -1)
                else:
                    _, emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                    emb_support = emb_support.reshape(1, test_n_support, -1)
                    labels_support = tile(labels_support, 1, dc_factor)

                if not opt.DC_query_inner:
                    emb_query, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                    emb_query = emb_query.reshape(1, test_n_query, -1)
                else:
                    _, emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                    emb_query = emb_query.reshape(1, test_n_query, -1)
                    labels_query = tile(labels_query, 1, dc_factor)

                if opt.algorithm_type == 'DiMP':
                    logit_query, inner_losses = cls_head(emb_query, emb_support, labels_support, opt.test_way,
                                                         int(emb_support.shape[1]/opt.test_way))

                    if opt.val_fusion_module == 1 or opt.val_fusion_module == 2:
                        #labels_query = labels_query.view(labels_query.shape[0], labels_query.shape[1]/dc_factor, -1)
                        labels_query = labels_query[:,range(0, labels_query.shape[1], dc_factor)]
                        logit_query[-1] = fusion_mod(logit_query[-1])

                    loss = x_entropy(logit_query[-1].reshape(-1, opt.test_way), labels_query.reshape(-1))
                    acc = count_accuracy(logit_query[-1].reshape(-1, opt.test_way), labels_query.reshape(-1))
                else:
                    # logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way,
                    #                        opt.val_shot * (opt.dropout_samples_per_class_val + 1))
                    logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way,
                                           int(emb_support.shape[1]/opt.test_way))
                    if opt.val_fusion_module == 1 or opt.val_fusion_module == 2:
                        labels_query = labels_query[:,range(0, labels_query.shape[1], dc_factor)]
                        logit_query[-1] = fusion_mod(logit_query[-1])
                    loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
                    acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))


                val_accuracies.append(acc.item())
                val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        # print("The sgd losses (val):")
        # for i_iter in range(len(inner_losses)):
        #     print("Iter " + str(i_iter) + ": " + str(inner_losses[i_iter]))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('Loss/val', val_loss_avg, epoch)

        # write loss to the tensorboard
        writer.add_scalar('Loss/classification', torch.tensor(0.0).cuda(), epoch)

        writer.add_scalar('Accuracy/train', train_acc_avg, epoch)
        writer.add_scalar('Accuracy/val', val_acc_avg, epoch)

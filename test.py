# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm
import multiprocessing

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.fusion import fuse_score, fuse_prob
from models.classification_heads import ClassificationHead
from models.WideResNet28_10_embedding import WideResNet

from utils import pprint, set_gpu, Timer, count_accuracy, log

import numpy as np
import os

from models.dimp_heads import  dimp_norm_init_shannon_hingeL2Loss


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


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'WideResNet_DC':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            if options.aws == 1:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=21).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=21).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            if options.aws == 1:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=8).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = WideResNet(28, widen_factor=10, dropRate=0.1, avgpool_param=8).cuda()
                network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
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
    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()    
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif opt.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif opt.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'FIML':
        cls_head = dimp_norm_init_shannon_hingeL2Loss(num_iter=options.steepest_descent_iter,
                                                      norm_feat=options.norm_feat,
                                                      entropy_weight=options.entropy_weight,
                                                      entropy_temp=options.entropy_temp,
                                                      learn_entropy_weights=options.learn_entropy_weights,
                                                      learn_entropy_temp=options.learn_entropy_temp,
                                                      learn_weights=options.learn_weights,
                                                      pos_weight=options.pos_weight, neg_weight=options.neg_weight,
                                                      learn_slope=options.learn_slope,
                                                      pos_lrelu_slope=options.pos_lrelu_slope,
                                                      neg_lrelu_slope=options.neg_lrelu_slope,
                                                      learn_spatial_weight=options.learn_inner_spatial_weight,
                                                      dc_factor=options.dc_factor)

    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    return (dataset_test, data_loader)

if __name__ == '__main__':

    # multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./experiments/exp_1/best_model.pth',
                            help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')

    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')

    parser.add_argument('--norm-feat', type=int, default=0,
                        help='to normalize the features or not. 1, 0')

    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='FIML',
                            help='choose which embedding network to use')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')

    parser.add_argument('--steepest-descent-iter', type=int, default=10, help='number of steepest descent iterations')


    parser.add_argument('--algorithm-type', type=str, default='DiMP',
                        help='Are we using the DiMP class of algos or other: DiMP, Other')

    parser.add_argument('--transductive-learning', type=str, default='True',
                        help='Employ transductive learning or not. True, False')

    parser.add_argument('--transductive-reg-weight', type=float, default=0,
                        help='Transductive Learning Weight')

    parser.add_argument('--smooth-label-base-learner', type=str, default='False',
                        help='whether to use the smooth labels inside the base learner or not')
    parser.add_argument('--smooth-label-factor-base-learner', type=float, default=0.0,
                        help='smooth factor for label smoothing in base learner')

    # aws stuff
    parser.add_argument('--aws', type=int, default=0,
                        help='are we running on aws')

    parser.add_argument('--entropy-weight', type=float, default=1e3, help='entropy weight')
    parser.add_argument('--entropy-temp', type=float, default=1, help='entropy temp')
    parser.add_argument('--learn-entropy-weights', type=str2bool, default=True, help='whether to learn entropy weights')
    parser.add_argument('--learn-entropy-temp', type=str2bool, default=True, help='whether to learn entropy temp')

    parser.add_argument('--pos-weight', type=float, default=1, help='pos example weight')
    parser.add_argument('--neg-weight', type=float, default=1, help='neg example weight')
    parser.add_argument('--pos-lrelu-slope', type=float, default=1.0, help='positive relu slope')
    parser.add_argument('--neg-lrelu-slope', type=float, default=1.0, help='negative relu slope')
    parser.add_argument('--learn-weights', type=str2bool, default=True, help='whether to learn pos neg weights')
    parser.add_argument('--learn-slope', type=str2bool, default=True, help='whether to learn relu slope')

    # dense classification
    # dense classification
    parser.add_argument('--DC-support', type=str2bool, default=True,
                        help='whether to use dense classification at support level')

    parser.add_argument('--DC-query-inner', type=str2bool, default=True,
                        help='whether to use dense classification at query level')

    parser.add_argument('--test-fusion-module', type=int, default=1,
                        help='0 is for the true query samples, 1 is for score fusion, 2 is for probability fusion')

    parser.add_argument('--learn-fusion-weight', type=str2bool, default=True,
                        help='whether to learn the fusion weights or not')

    parser.add_argument('--learn-inner-spatial-weight', type=str2bool, default=True,
                        help='whether to learn the fusion weights or not')

    opt = parser.parse_args()

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
    (dataset_test, data_loader) = get_dataset(opt)

    # dloader_test = data_loader(
    #     dataset=dataset_test,
    #     nKnovel=opt.way,
    #     nKbase=0,
    #     nExemplars=opt.shot,  # num training examples per novel category
    #     nTestNovel=opt.query * opt.way,  # num test examples for all the novel categories
    #     nTestBase=0,  # num test examples for all the base categories
    #     batch_size=1,
    #     num_workers=1,
    #     epoch_size=opt.episode,  # num of batches per epoch
    # )

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    # if opt.aws == 1:
    #     set_gpu(opt.gpu)

    test_loader = dloader_test.get_dataloader()

    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model(opt)



    if opt.test_fusion_module == 1:
        fusion_mod = fuse_score(dc_factor=dc_factor, weight_learn=opt.learn_fusion_weight).cuda()
    elif opt.test_fusion_module == 2:
        fusion_mod = fuse_prob(dc_factor=dc_factor, weight_learn=opt.learn_fusion_weight).cuda()
    else:
        fusion_mod = fuse_score(dc_factor=dc_factor, weight_learn=opt.learn_fusion_weight).cuda()

    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()



    # Evaluate on test set
    test_accuracies = []

    torch.set_grad_enabled(False)
    # for i, batch in enumerate(tqdm(dloader_test()), 1):
    for i, batch in enumerate(dloader_test(), 1):
    # for i, batch in enumerate(test_loader):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        if opt.DC_support:
            n_support = opt.way * opt.shot * dc_factor
        else:
            n_support = opt.way * opt.shot

        if opt.DC_query_inner:
            n_query = opt.way * opt.query * dc_factor
        else:
            n_query = opt.way * opt.query

        # emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
        # emb_support = emb_support.reshape(1, n_support, -1)

        if not opt.DC_support:
            emb_support, _ = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, n_support, -1)
        else:
            _, emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, n_support, -1)
            labels_support = tile(labels_support, 1, dc_factor)

        if not opt.DC_query_inner:
            emb_query, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, n_query, -1)
        else:
            _, emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, n_query, -1)
            labels_query = tile(labels_query, 1, dc_factor)

        if opt.head == 'SVM':
            #logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot*(opt.dropout_samples_per_class+1), maxIter=3)
            logits = cls_head(emb_query, emb_support, labels_support, opt.way,
                              int(emb_support.shape[1]/opt.way), maxIter=3)
            acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        else:
            if opt.algorithm_type == 'DiMP':
                logits, _ = cls_head(emb_query, emb_support, labels_support, opt.way, int(emb_support.shape[1]/opt.way))

                if opt.test_fusion_module == 1 or opt.test_fusion_module == 2:
                    #labels_query = labels_query.view(labels_query.shape[0], labels_query.shape[1] / dc_factor, -1)
                    labels_query = labels_query[:,range(0, labels_query.shape[1], dc_factor)]
                    logits[-1] = fusion_mod(logits[-1])

                acc = count_accuracy(logits[-1].reshape(-1, opt.way), labels_query.reshape(-1))
            else:
                #logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot*(opt.dropout_samples_per_class+1))
                logits = cls_head(emb_query, emb_support, labels_support, opt.way,
                                  int(emb_support.shape[1]/opt.way))
                acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))

        # acc = count_accuracy(logits[-1].reshape(-1, opt.way), labels_query.reshape(-1))
        # acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)
        
        if i % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))

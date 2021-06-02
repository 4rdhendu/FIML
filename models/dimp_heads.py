import torch
# from torch.autograd import Variable
import torch.nn as nn

import torch
import torch.nn as nn
import math
from libs import TensorList
from models.classification_heads import one_hot


def classify(features, weights):
    # features: (n_tasks, n_samples, d)
    # weights: (n_tasks, n_way, d)
    # o/p: (n_tasks, n_samples, n_way)
    return torch.matmul(features, weights.permute(0, 2, 1))



class LeakyReluPar(nn.Module):
    r"""LeakyRelu parametric activation
    """
    def forward(self, x, a):
        return (1.0 - a)/2.0 * torch.abs(x) + (1.0 + a)/2.0 * x

class LeakyReluParDeriv(nn.Module):
    r"""Derivative of the LeakyRelu parametric activation, wrt x.
    """
    def forward(self, x, a):
        return (1.0 - a)/2.0 * torch.sign(x.detach()) + (1.0 + a)/2.0

class ResLabelLearnSoftHinge(nn.Module):
    """ |s - y|^2  ;  y = yp, yn  (pos_label or neg_label)
        for negative class:  |max(s - yn, 0)|^2
        for positive class:  |min(s, yp) - yp|^2 = |max(yp - s, 0) |^2
        for both: |v * max(sign(y)*(y - s), 0)|^2      |v * max(sign(y)*(y - s), slope(y))|^2
    """

    def __init__(self, init_pos=5.0, init_neg=-5.0, init_reg=0, learn_reg=True,
                 pos_weight=1.0, neg_weight=1.0, learn_weights=False,
                 pos_lrelu_slope=1.0, neg_lrelu_slope=1.0, learn_slope=False, learn_spatial_weight=False, dc_factor=0):
        super().__init__()
        self.pos_label = nn.Parameter(torch.Tensor([init_pos]))
        self.neg_label = nn.Parameter(torch.Tensor([init_neg]))

        self.weight_reg = torch.Tensor([init_reg])
        if learn_reg:
            self.weight_reg = nn.Parameter(self.weight_reg)

        self.pos_weight = torch.Tensor([pos_weight])
        self.neg_weight = torch.Tensor([neg_weight])
        if learn_weights:
            self.pos_weight = nn.Parameter(self.pos_weight)
            self.neg_weight = nn.Parameter(self.neg_weight)

        self.pos_lrelu_slope = torch.Tensor([pos_lrelu_slope])
        self.neg_lrelu_slope = torch.Tensor([neg_lrelu_slope])
        if learn_slope:
            self.pos_lrelu_slope = nn.Parameter(self.pos_lrelu_slope)
            self.neg_lrelu_slope = nn.Parameter(self.neg_lrelu_slope)

        self.dc_factor = dc_factor
        if learn_spatial_weight:
            self.wt_spatial = torch.ones((dc_factor, 1)) * (1 / dc_factor)
        else:
            self.wt_spatial = nn.Parameter(torch.Tensor(torch.ones((dc_factor,1))*(1/dc_factor)))
        self.res_activation = LeakyReluPar()
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, weights, support, support_labels, n_way, query):

        device = support.device
        wt_norm_spat = self.sm(self.wt_spatial).repeat(int(support.shape[1]/self.dc_factor), 1)
        weights = weights[0]

        tasks_per_batch = support.size(0)
        n_support = support.size(1)

        one_hot_label = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)  # (tasks_per_batch * n_support, n_way)

        one_hot_label = one_hot_label.reshape(tasks_per_batch, n_support, n_way)
        label_sgn = 2*one_hot_label - 1

        label = self.pos_label.to(device)*one_hot_label + (1-one_hot_label)*self.neg_label.to(device)
        label_weights = self.pos_weight.to(device)*one_hot_label + (1-one_hot_label)*self.neg_weight.to(device)
        label_lrelu_slope = self.pos_lrelu_slope.to(device)*one_hot_label + (1-one_hot_label)*self.neg_lrelu_slope.to(device)

        scores = classify(support, weights)

        data_residual = wt_norm_spat.to(device) * label_weights.to(device) * self.res_activation(label_sgn * (label - scores), label_lrelu_slope)

        reg_residual = self.weight_reg.to(weights.device)*weights
        return TensorList([data_residual, reg_residual])


class WeightInitializerNorm(nn.Module):
    def __init__(self, init_pos=1.0, init_neg=0.0):
        super().__init__()
        self.target_pos = nn.Parameter(torch.Tensor([init_pos]))
        self.target_neg = nn.Parameter(torch.Tensor([init_neg]))

    def forward(self, support, support_labels, n_way):

        tasks_per_batch = support.size(0)
        n_support = support.size(1)
        one_hot_label = one_hot(support_labels.view(-1), n_way).view(tasks_per_batch, n_support, n_way).permute(0,2,1)

        pos_feat = torch.matmul(one_hot_label, support)
        neg_feat = support.sum(dim=1, keepdim=True) - pos_feat

        k_shot = one_hot_label.sum(dim=-1).view(tasks_per_batch, n_way, 1)
        pos_feat = pos_feat / k_shot
        neg_feat = neg_feat / (n_support - k_shot)

        ff = (pos_feat * pos_feat).sum(dim=2, keepdim=True)
        bb = (neg_feat * neg_feat).sum(dim=2, keepdim=True)
        fb = (pos_feat * neg_feat).sum(dim=2, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_pos.to(bb.device) * bb - self.target_neg.to(fb.device) * fb
        bg_scale = self.target_pos.to(fb.device) * fb - self.target_neg.to(ff.device) * ff
        weights = (fg_scale * pos_feat - bg_scale * neg_feat) / den

        return weights




class GNSteepestDescentShannonEntropy(nn.Module):
    def __init__(self, residual_module, num_iter=1, compute_losses=True, detach_length=float('Inf'),
                 parameter_batch_dim=0, residual_batch_dim=0, entropy_weight=0.0, entropy_temp=1.0, learn_entropy_weight=False,
                 learn_entropy_temp=False, steplength_reg=0.05):
        super().__init__()

        self.residual_module = residual_module
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self._parameter_batch_dim = parameter_batch_dim
        self._residual_batch_dim = residual_batch_dim
        self.steplength_reg = steplength_reg

        if learn_entropy_weight:
            self.entropy_weight = nn.Parameter(torch.Tensor([entropy_weight]))
        else:
            self.entropy_weight = torch.Tensor([entropy_weight])

        if learn_entropy_temp:
            self.entropy_temp = nn.Parameter(torch.Tensor([entropy_temp]))
        else:
            self.entropy_temp = torch.Tensor([entropy_temp])


    def _sqr_norm(self, x: TensorList, batch_dim=0):
        sum_keep_batch_dim = lambda e: e.sum(dim=[d for d in range(e.dim()) if d != batch_dim])
        return sum((x * x).apply(sum_keep_batch_dim))

    def _compute_loss(self, res, scores_q):
        losses = 0.5*(res * res).sum()
        entropy_weight = self.entropy_weight / scores_q.shape[1]
        prob_q = torch.softmax(scores_q, dim=-1)
        ent_loss = entropy_weight * (torch.logsumexp(scores_q, dim=-1) - torch.sum(scores_q * prob_q, dim=-1)).sum()
        losses.append(ent_loss)
        return losses

    def forward(self, meta_parameter: TensorList, num_iter=None,  *args, **kwargs):
        # Make sure grad is enabled
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        num_iter = self.num_iter if num_iter is None else num_iter

        query = kwargs['query']
        n_query = query.shape[1]
        entropy_weight = self.entropy_weight / n_query

        # def _compute_loss(res):
        #     loss = sum((res * res).sum()) / sum(res.numel())
        #     if not torch_grad_enabled:
        #         loss.detach_()
        #     return loss


        meta_parameter_iterates = [meta_parameter]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()

            meta_parameter.requires_grad_(True)

            # Compute residual vector
            r = self.residual_module(meta_parameter, *args, **kwargs)

            # Compute gradient of loss
            u = r.clone()
            weight_res_grad = TensorList(torch.autograd.grad(r, meta_parameter, u, create_graph=True))

            # Compute query scores
            scores_q = self.entropy_temp.to(query.device) * classify(query, meta_parameter[0])
            scores_q_softmax = torch.softmax(scores_q, dim=-1)
            dLds = entropy_weight.to(query.device) * scores_q_softmax * (torch.sum(scores_q*scores_q_softmax, dim=-1, keepdim=True) - scores_q)

            # Compute gradient of loss
            weights_entropy_grad = TensorList(torch.autograd.grad(scores_q, meta_parameter, dLds, create_graph=True))

            # Total gradient
            weights_grad = weight_res_grad + weights_entropy_grad

            # Multiply gradient with Jacobian
            h = TensorList(torch.autograd.grad(weight_res_grad, u, weights_grad, create_graph=True))


            # Multiply gradient with Jacobian for query entropy
            scores_q_grad = torch.autograd.grad(weights_entropy_grad, dLds, weights_grad, create_graph=True)[0]

            # Get hessian product for query entropy
            sm_scores_grad = scores_q_softmax * scores_q_grad
            hes_scores_grad = sm_scores_grad - scores_q_softmax * torch.sum(sm_scores_grad, dim=-1, keepdim=True)
            grad_hes_grad_ent = (scores_q_grad * hes_scores_grad).sum(dim=-1).clamp(min=0)
            grad_hes_grad_ent = (entropy_weight.to(query.device) * grad_hes_grad_ent).sum(dim=-1)    # should have shape [batch_dim]

            # Compute squared norms
            ip_gg = self._sqr_norm(weights_grad, batch_dim=self._parameter_batch_dim)
            ip_hh = self._sqr_norm(h, batch_dim=self._residual_batch_dim) + grad_hes_grad_ent

            # Compute step length
            alpha = ip_gg / (ip_hh.clamp(1e-8)+self.steplength_reg*ip_gg)

            # Multiply gradient with step length
            step = weights_grad.apply(lambda e: alpha.view([-1 if d==self._parameter_batch_dim else 1 for d in range(e.dim())]) * e)

            if self.compute_losses:
                losses.append(self._compute_loss(r, scores_q))

            # Add step to parameter
            meta_parameter = meta_parameter - step

            if not torch_grad_enabled:
                meta_parameter.detach_()

            meta_parameter_iterates.append(meta_parameter)


        if self.compute_losses:
            losses.append(self._compute_loss(self.residual_module(meta_parameter, *args, **kwargs),
                                             self.entropy_temp.to(query.device) * classify(query, meta_parameter[0])))

        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()

        # Reset the grad enabled flag
        torch.set_grad_enabled(torch_grad_enabled)

        return meta_parameter, meta_parameter_iterates, losses

class DiMPHead(nn.Module):
    # softmax required the norm_feat to be 1
    def __init__(self, weight_initializer, weight_optimizer, norm_feat=0):
        super().__init__()
        self.weight_initializer = weight_initializer
        self.weight_optimizer = weight_optimizer
        self.norm_feat = norm_feat

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        weights = self.weight_initializer(support, support_labels, n_way)

        if self.norm_feat==1:
            query = query/torch.sqrt(torch.sum(query*query, dim=2)).unsqueeze(2)
            support = support/torch.sqrt(torch.sum(support*support, dim=2)).unsqueeze(2)
        weights, weights_iterate, losses = self.weight_optimizer(TensorList([weights]), support=support,
                                                                 support_labels=support_labels, n_way=n_way, query=query, **kwargs)

        class_scores = [classify(query, w[0]) for w in weights_iterate]
        return class_scores, losses

# shannon entropy transductive loss
def dimp_norm_init_shannon_hingeL2Loss(num_iter=5, init_pos=5.0, init_neg=-5.0, pos_lrelu_slope = 1.0, neg_lrelu_slope = 1.0,
                                       norm_feat=0, pos_weight=1.0, neg_weight=1.0, learn_slope = False,
                                       learn_weights=False, entropy_weight=1e3, entropy_temp=1, learn_entropy_weights=False,
                                       learn_entropy_temp=False, learn_spatial_weight=False, dc_factor=0):
    initializer = WeightInitializerNorm(init_pos=init_pos, init_neg=init_neg)
    residual_module = ResLabelLearnSoftHinge(init_pos=init_pos, init_neg=init_neg, learn_weights=learn_weights,
                                             learn_slope=learn_slope, pos_weight=pos_weight, neg_weight=neg_weight,
                                             pos_lrelu_slope=pos_lrelu_slope, neg_lrelu_slope=neg_lrelu_slope,
                                             learn_spatial_weight=learn_spatial_weight, dc_factor=dc_factor)

    optimizer = GNSteepestDescentShannonEntropy(residual_module, num_iter, entropy_weight=entropy_weight, entropy_temp=entropy_temp,
                                                        learn_entropy_weight=learn_entropy_weights, learn_entropy_temp=learn_entropy_temp)
    cls_head = DiMPHead(initializer, optimizer, norm_feat)
    return cls_head
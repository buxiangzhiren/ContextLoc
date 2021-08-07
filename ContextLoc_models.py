import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np
from numpy.random import randint
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight1 = Parameter(torch.Tensor(in_features, out_features))
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # support1 = torch.mm(input, self.weight1)
        output = torch.mm(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x

class LGNet(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=True):
        super(LGNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(in_features, out_features))
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.dropout = dropout
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, x_child, adj):
        support = torch.mm(x, self.weight)
        support1 = torch.mm(x_child, self.weight1)
        output = support + torch.mm(adj, support1)
        if self.bias is not None:
            output = output + self.bias
        x = F.relu(output)
        return x



class ContextLoc(torch.nn.Module):
    def __init__(self, model_configs, graph_configs, dataset_configs, test_mode=False):
        super(ContextLoc, self).__init__()

        self.num_segments = model_configs['num_segments']
        self.ft_test_path = dataset_configs['test_ft_path']
        self.ft_train_path = dataset_configs['train_ft_path']
        self.num_class = model_configs['num_class']
        self.adj_num = graph_configs['adj_num']
        self.child_num = graph_configs['child_num']
        self.child_iou_num = graph_configs['iou_num']
        self.child_dis_num = graph_configs['dis_num']
        self.dropout = model_configs['dropout']
        self.test_mode = test_mode
        self.act_feat_dim = model_configs['act_feat_dim']
        self.comp_feat_dim = model_configs['comp_feat_dim']

        self._prepare_pgcn()
        self.Act_GCN = GCN(self.act_feat_dim, 512, self.act_feat_dim, dropout=model_configs['gcn_dropout'])
        self.Comp_GCN = GCN(self.comp_feat_dim, 512, self.comp_feat_dim, dropout=model_configs['gcn_dropout'])
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.LGNet = LGNet(1024, 512, dropout=model_configs['gcn_dropout'])
        self.LGNet_G = LGNet(1024, 512, dropout=model_configs['gcn_dropout'])
        self.BN_A = nn.BatchNorm1d(self.act_feat_dim)
        self.BN_C = nn.BatchNorm1d(self.comp_feat_dim)



    def mI3D_Pooling(self, v_list, prop_indices, v_index, n_frame, n_seg=1):

        if self.training:
            ft_path = self.ft_train_path
        else:
            ft_path = self.ft_test_path
        if self.test_mode:
            ft_path = self.ft_test_path
        vid_full_name = v_list[v_index].id
        vid = vid_full_name.split('/')[-1]
        ft_tensor = torch.load(os.path.join(ft_path, vid)).float().cuda()
        global_ft = torch.max(ft_tensor, 0)[0].view(1, -1)
        fts_all_act = []
        fts_all_comp = []


        for prop in prop_indices:
            act_s = prop[0]
            act_e = prop[1]
            comp_s = prop[2]
            comp_e = prop[3]

            start_ft = self.feature_pooling(comp_s, act_s, n_seg, ft_tensor, global_ft)
            end_ft = self.feature_pooling(act_e, comp_e, n_seg, ft_tensor, global_ft)
            act_ft = self.feature_pooling(act_s, act_e, n_seg, ft_tensor, global_ft)
            comp_ft = [start_ft, act_ft, end_ft]
            comp_ft = torch.cat(comp_ft, dim=0)

            fts_all_act.append(act_ft)
            fts_all_comp.append(comp_ft)


        fts_all_act = torch.stack(fts_all_act)
        fts_all_comp = torch.stack(fts_all_comp)

        return fts_all_act, fts_all_comp

    def feature_pooling(self, start_ind, end_ind, n_seg, ft_tensor, global_ft):
        # for turn
        interval = 8
        clip_length = 64

        fts = []
        fts_all = []

        offsets, average_duration = self.sample_indices(start_ind, end_ind, n_seg)

        ft_num = ft_tensor.size()[0]

        for off in offsets:

            fts = []

            start_unit = int(min(ft_num - 1, np.floor(float(start_ind + off) / interval)))
            end_unit = int(min(ft_num - 2, np.ceil(float(end_ind - clip_length) / interval)))

            if start_unit < end_unit:
                s_pro_ft = ft_tensor[start_unit: end_unit + 1, :].cuda()
                pro_ft = torch.max(ft_tensor[start_unit: end_unit+1, :], 0)[0].view(1, -1)
                s_pro_ft_p = torch.cat((s_pro_ft, global_ft), dim=0)
                s_pro_ft_g = torch.cat((s_pro_ft, pro_ft), dim=0)
                pro_cos_sim_mat = F.cosine_similarity(pro_ft, s_pro_ft_p, dim=1, eps=1e-6)
                pro_ft_adj = F.relu(pro_cos_sim_mat.view(1, -1))
                pro_ft_adj = pro_ft_adj / torch.sum(pro_ft_adj)
                pro_gcn_ft = self.LGNet(pro_ft, s_pro_ft_p, pro_ft_adj)
                glo_cos_sim_mat = F.cosine_similarity(global_ft, s_pro_ft_g, dim=1, eps=1e-6)
                glo_ft_adj = F.relu(glo_cos_sim_mat.view(1, -1))
                glo_ft_adj = glo_ft_adj / torch.sum(glo_ft_adj)
                glo_gcn_ft = self.LGNet_G(global_ft, s_pro_ft_g, glo_ft_adj)
                pro_gcn_ft = torch.cat((pro_gcn_ft, glo_gcn_ft), dim=-1)
                fts.append(pro_gcn_ft)
                # fts.append(torch.max(pro_gcn_ft + s_pro_ft, 0)[0])
                # fts.append(torch.max(ft_tensor[start_unit: end_unit+1, :], 0)[0])
            else:
                pro_ft = ft_tensor[start_unit].view(1, -1)
                cos_sim_mat = F.cosine_similarity(pro_ft, global_ft, dim=1, eps=1e-6)
                ft_adj = F.relu(cos_sim_mat.view(1, -1))
                pro_ft_t = self.LGNet(pro_ft, global_ft, ft_adj)
                global_ft_t = self.LGNet_G(global_ft, pro_ft, ft_adj)
                pro_gcn_ft = torch.cat((pro_ft_t, global_ft_t), dim=-1)
                fts.append(pro_gcn_ft)
            fts_all.append(fts[0])

        fts_all = torch.stack(fts_all).cuda()

        return fts_all.squeeze()

    def sample_indices(self, start, end, num_seg):
        """
        :param record: VideoRecord
        :return: list
        """
        valid_length = end - start
        v_l_t = valid_length.clone().detach().cpu()
        average_duration = (valid_length + 1) // num_seg
        a_d_t = average_duration.clone().detach().cpu()
        if average_duration > 0:
            # normal cases
            offsets = np.multiply(list(range(num_seg)), a_d_t)
        elif valid_length > num_seg:
            offsets = np.sort(randint(v_l_t, size=num_seg))
        else:
            offsets = np.zeros((num_seg,))

        return offsets, average_duration

    def _prepare_pgcn(self):

        self.activity_fc = nn.Linear(self.act_feat_dim * 2, self.num_class + 1)
        self.completeness_fc = nn.Linear(self.comp_feat_dim * 2, self.num_class)
        self.regressor_fc = nn.Linear(self.comp_feat_dim * 2, 2 * self.num_class)

        nn.init.normal_(self.activity_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.activity_fc.bias.data, 0)
        nn.init.normal_(self.completeness_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.completeness_fc.bias.data, 0)
        nn.init.normal_(self.regressor_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.regressor_fc.bias.data, 0)


    def train(self, mode=True):

        super(ContextLoc, self).train(mode)


    def get_optim_policies(self):

        normal_weight = []
        normal_bias = []
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, GraphConvolution):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, LGNet):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                if len(ps) == 3:
                    normal_bias.append(ps[2])
            elif isinstance(m, nn.BatchNorm1d):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ]

    def forward(self, v_list, out_prop_ind, v_index, num_frames, target, reg_target, prop_type, input = None):
        if not self.test_mode:
            activity_fts_t = []
            completeness_fts_t = []
            a = torch.stack(out_prop_ind)
            for i in range(0, len(v_index)):
                act_prop_ft, comp_prop_ft = self.mI3D_Pooling(v_list, a[:, i, :], v_index[i], num_frames[i])
                activity_fts_t.append(act_prop_ft)
                completeness_fts_t.append(comp_prop_ft)
            activity_fts = torch.stack(activity_fts_t).cuda()
            completeness_fts = torch.stack(completeness_fts_t).cuda()
            return self.train_forward((activity_fts, completeness_fts), target, reg_target, prop_type)
        else:
            return self.test_forward(input)


    def train_forward(self, input , target, reg_target, prop_type):

        activity_fts = input[0]
        completeness_fts = input[1]

        batch_size = activity_fts.size()[0]

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous()
        comp_ft_mat = completeness_fts.view(-1, self.comp_feat_dim).contiguous()
        act_ft_mat = self.BN_A(act_ft_mat)
        comp_ft_mat = self.BN_C(comp_ft_mat)


        # act cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        act_cos_sim_mat = dot_product_mat / len_mat

        # comp cosine similarity
        dot_product_mat = torch.mm(comp_ft_mat, torch.transpose(comp_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(comp_ft_mat * comp_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        comp_cos_sim_mat = dot_product_mat / len_mat

        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num)
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0])
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)):
            mask_mat_var[row * self.adj_num : (row + 1) * self.adj_num, row * self.adj_num : (row + 1) * self.adj_num] \
                = mask

        act_adj_mat = mask_mat_var * act_cos_sim_mat
        comp_adj_mat = mask_mat_var * comp_cos_sim_mat

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(act_ft_mat, act_adj_mat)
        comp_gcn_ft = self.Comp_GCN(comp_ft_mat, comp_adj_mat)

        out_act_fts = torch.cat((act_gcn_ft, act_ft_mat), dim=-1)
        act_fts = out_act_fts[:-1: self.adj_num, :]
        act_fts = self.dropout_layer(act_fts)

        out_comp_fts = torch.cat((comp_gcn_ft, comp_ft_mat), dim=-1)
        comp_fts = out_comp_fts[:-1: self.adj_num, :]

        raw_act_fc = self.activity_fc(act_fts)
        raw_comp_fc = self.completeness_fc(comp_fts)

        # keep 7 proposal to calculate completeness
        raw_comp_fc = raw_comp_fc.view(batch_size, -1, raw_comp_fc.size()[-1])[:, :-1, :].contiguous()
        raw_comp_fc = raw_comp_fc.view(-1, raw_comp_fc.size()[-1])
        comp_target = target.view(batch_size, -1, self.adj_num)[:, :-1, :].contiguous().view(-1).data
        comp_target = comp_target[0: -1: self.adj_num].contiguous()

        # keep the target proposal
        type_data = prop_type.view(-1).data
        type_data = type_data[0: -1: self.adj_num]
        target = target.view(-1)
        target = target[0: -1: self.adj_num]

        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()

        reg_target = reg_target.view(-1, 2)
        reg_target = reg_target[0: -1: self.adj_num]
        reg_indexer = (type_data == 0).nonzero().squeeze()
        raw_regress_fc = self.regressor_fc(comp_fts).view(-1, self.completeness_fc.out_features, 2).contiguous()

        return raw_act_fc[act_indexer, :], target[act_indexer], type_data[act_indexer], \
               raw_comp_fc, comp_target, \
              raw_regress_fc[reg_indexer, :, :], target[reg_indexer], reg_target[reg_indexer, :]

    def test_forward(self, input):

        activity_fts = input[0]
        completeness_fts = input[1]
        batch_size = activity_fts.size()[0]

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous()
        comp_ft_mat = completeness_fts.view(-1, self.comp_feat_dim).contiguous()

        act_ft_mat = self.BN_A(act_ft_mat)
        comp_ft_mat = self.BN_C(comp_ft_mat)


        # act cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        act_cos_sim_mat = dot_product_mat / len_mat

        # comp cosine similarity
        dot_product_mat = torch.mm(comp_ft_mat, torch.transpose(comp_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(comp_ft_mat * comp_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        comp_cos_sim_mat = dot_product_mat / len_mat

        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num)
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0])
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)):
            mask_mat_var[row * self.adj_num: (row + 1) * self.adj_num, row * self.adj_num: (row + 1) * self.adj_num] \
                = mask

        act_adj_mat = mask_mat_var * act_cos_sim_mat
        comp_adj_mat = mask_mat_var * comp_cos_sim_mat

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(act_ft_mat, act_adj_mat)
        comp_gcn_ft = self.Comp_GCN(comp_ft_mat, comp_adj_mat)

        out_act_fts = torch.cat((act_gcn_ft, act_ft_mat), dim=-1)

        act_fts = out_act_fts[:-1: self.adj_num, :]

        out_comp_fts = torch.cat((comp_gcn_ft, comp_ft_mat), dim=-1)

        comp_fts = out_comp_fts[:-1: self.adj_num, :]

        raw_act_fc = self.activity_fc(act_fts)
        raw_comp_fc = self.completeness_fc(comp_fts)

        raw_regress_fc = self.regressor_fc(comp_fts).view(-1, self.completeness_fc.out_features * 2).contiguous()

        return raw_act_fc, raw_comp_fc, raw_regress_fc



#expert_sum 


from torch import nn
from transformers.activations import get_activation

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import copy
import typing
#import ipdb
#from transformers.models.mixture_of_experts import MoE
from typing import List, Optional, Tuple, Union
################################################################### AdaMix + AdaKron

class MixtureSoup(torch.nn.Module):

    def __init__(self, expert, num_local_experts, inference_level=0):
        super(MixtureSoup, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        self.inference_level = inference_level
        self.expert_score_weight = torch.nn.Parameter(torch.rand(self.num_local_experts), requires_grad=False)

    def get_expert_by_idx(self, idx):
        return self.deepspeed_experts[idx]

    def expert_soup_forward(self, input):
        output = F.linear(input,
                          self.parameter_dict["weight"],
                          self.parameter_dict["bias"])
        return output

    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight)
        self.parameter_dict = {"weight": 0, "bias": 0}
        for idx in range(self.num_local_experts):
            single_expert = self.deepspeed_experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)
                else:
                    p_name = "bias"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)

    def forward(self, *input: Tensor):
        expert_output = None
        if self.deepspeed_experts[0].training:
            #ipdb.set_trace()
            expert_idx = torch.randint(low=0, high=self.num_local_experts, size=(1,)).item()  # selected expert
            if self.expert_score_weight.requires_grad:
                self.expert_soup()
                expert_output = self.expert_soup_forward(input[0])
            else:
                expert_output = self.get_expert_by_idx(expert_idx)(input[0])

        else:
            if self.inference_level != 3:
                result = []
                for expert_idx in range(self.num_local_experts):
                    temp = self.get_expert_by_idx(expert_idx)(input[0])
                    result.append(temp)
                result = torch.stack(result, dim=0)

                if self.inference_level == 0:  # token level
                    mask = torch.randint(0, self.num_local_experts,
                                         size=(result.size(1), result.size(2)), device=result.device)
                    for i in range(self.num_local_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1)
                    expert_output = result.sum(0)
                elif self.inference_level == 1:  # sentence level
                    mask = torch.randint(0, self.num_local_experts,
                                         size=(result.size(1),), device=result.device)
                    for i in range(self.num_local_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                    expert_output = result.sum(0)

            elif self.inference_level == 3:
                self.expert_soup()
                expert_output = self.expert_soup_forward(input[0])

        return expert_output


class ExpertSoup(nn.Module):
    def __init__(self, dim, r, act=None, num_expert=1, sharing_down=0, sharing_up=0):
        super().__init__()

        self.act = act
        # if sharing_down == 1:
        #     self.MoA_A = MixtureSoup(nn.Linear(dim, r), 1, inference_level)
        # else:
        self.MoA_A = MixtureSoup(nn.Linear(dim, 4), num_expert, 3)
        self.MoA_A1 = MixtureSoup(nn.Linear(dim, int(r/4)), 1, 3)
        if act is not None:
            self.act = get_activation(act)

        # if sharing_up == 1:
        #     self.MoA_B = MixtureSoup(nn.Linear(r, dim), 1, inference_level)
        # else:
        self.MoA_B = MixtureSoup(nn.Linear(r, dim), 1, 3)
        #self.MoA_B = MixtureSoup(nn.Linear(r, dim_expert), num_expert, inference_level)
        #self.MoA_B1 = MixtureSoup(nn.Linear(r, int(dim/dim_expert)), 1, inference_level)

    def kronecker(self,output1,output2):
        siz1 = torch.Size(torch.tensor(output1.shape[-1:]) * torch.tensor(output2.shape[-1:]))
        res = output1.unsqueeze(-1) * output2.unsqueeze(-2) 

        siz0 = res.shape[:-2]
        result = res.reshape(siz0 + siz1)

        return result

    def forward(self, x, residual):
        result1 = self.MoA_A(x)
        result2 = self.MoA_A1(x)
        result = self.kronecker(result2, result1)       #+ self.kronecker(result1, result2)
        if self.act is not None:
            result = self.act(result)
        result = self.MoA_B(result)
        #result1 = self.MoA_B(result)
        #result2 = self.MoA_B1(result)
        #result = self.kronecker(result2, result1)
        return result + residual
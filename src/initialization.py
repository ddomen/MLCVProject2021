import math

import torch
from torch import nn


def normal(mean, std, seed=None):
    def __normal(tensor):
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.normal_(tensor, mean, std)
    return __normal


def uniform(a, b, seed=None):
    def __uniform(tensor):
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.uniform_(tensor, a, b)
    return __uniform


def variance_scaling_fan_in(scale, seed=None):
    def __variance_scaling_fan_in(tensor):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = math.sqrt(scale / float(fan_in))
        a = math.sqrt(3.0 * scale) * std
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init._no_grad_uniform_(tensor, -a, a)
    return __variance_scaling_fan_in


def truncated_normal_(mean, std, seed=None):
    def __truncated_normal(tensor):
        size = tensor.shape
        if seed is not None:
            torch.manual_seed(seed)
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        return tensor.data.mul_(std).add_(mean)
    return __truncated_normal


def lecun_uniform(seed=None):
    def __lecun_uniform(tensor):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(3 / fan_in)
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.uniform_(tensor, -scale, scale)
    return __lecun_uniform


def glorot_uniform(seed=None):
    def __glorot_uniform(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(6 / (fan_in + fan_out))
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.uniform_(tensor, -scale, scale)
    return __glorot_uniform


def glorot_normal(seed=None):
    def __glorot_normal(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(2 / (fan_in + fan_out))
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.normal_(tensor, 0, scale)
    return __glorot_normal


def he_normal(seed=None):
    def __he_normal(tensor):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(2 / fan_in)
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.normal_(tensor, 0, scale)
    return __he_normal


def he_uniform(seed=None):
    def __he_uniform(tensor):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(6 / fan_in)
        if seed is not None:
            torch.manual_seed(seed)
        return nn.init.uniform_(tensor, -scale, scale)
    return __he_uniform

"""
Helper class for profiling NN models

This code is from Sergei Belousov

"""
import random
import numpy as np
import torch
from torch import nn


class ModelProfiler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.flops = 0
        self.units = {'K': 10.**3, 'M': 10.**6, 'G': 10.**9}

    def forward(self, *args, **kwargs):
        self.flops = 0
        self._init_hooks()
        output = self.model(*args, **kwargs)
        self._remove_hooks()
        return output

    def get_flops(self, units='G'):
        return self.flops / self.units[units]

    def get_params(self, units='G'):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if units is not None:
            params = params / self.units[units]
        return params

    def _remove_hooks(self):
        if self.hooks is not None:
            for hook in self.hooks:
                hook.remove()
        self.hooks = None

    def _init_hooks(self):
        self.hooks = []

        def hook_compute_flop(m, input, output):
            self.flops += m.weight.size()[1:].numel() * output.size()[1:].numel()

        def add_hooks(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(hook_compute_flop))

        self.model.apply(add_hooks)

def profile_model(model, input_size, cuda):
    profiler = ModelProfiler(model)
    var = torch.zeros(input_size)
    if cuda:
        var = var.cuda()
    profiler(var)
    print("FLOPs: {0:.5}; #Params: {1:.5}".format(profiler.get_flops('G'), profiler.get_params('M')))


def main():
    profile_model(vgg16, input_size=(1, 3, 224, 224), cuda=cuda)


if __name__ == "__main__":
    main()    

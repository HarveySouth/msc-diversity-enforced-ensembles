import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import stack_module_state
from torch import vmap
from torch.func import functional_call
import copy


def torch_MSE_combiner(ensemble_output, axis=0):
    return torch.mean(ensemble_output, axis=axis)

def torch_MCCE_logit_combiner(ensemble_output, axis=0): # softmax(mean(logits))) == normalzied_geometric_mean(softmax(logits))
    combined_logits = torch.mean(ensemble_output, axis=axis)
    return combined_logits

def norm_geo_mean(member_output): # take softmax (probabilistic) outputs from members and combine into centroid ensemble output 

    # mem out is shape, member, sample, class
    len_out = len(member_output)
    ens_out = torch.prod(member_output, dim=0)

    # print(ens_out)

    ens_out = torch.pow(ens_out, 1/len_out)

    # print(ens_out)
    
    Z_inverse = 1 / torch.sum(ens_out, dim=-1)

    # print(Z_inverse)

    normed_out = Z_inverse.unsqueeze(1) * ens_out #unsqueeze for correct broadcasting

    # print(normed_out)

    return normed_out



class Ensemble_Runner():
    def __init__(self, exmaple_model, combiner_rule):
        self.metamodel = copy.deepcopy(exmaple_model).to('meta')
        self.combiner_rule = combiner_rule
        
    def forward_through_metamodel(self, params, buffers, x):
        return functional_call(self.metamodel, (params, buffers), (x,))
    
    def forward(self, x, ensemble):
        params, buffers = stack_module_state(nn.ModuleList(ensemble))

        member_output = vmap(self.forward_through_metamodel)(params, buffers, x.repeat(len(ensemble), 1, 1))

        # print(member_output.shape)

        ensemble_output = self.combiner_rule(member_output)

        # print(ensemble_output.shape)
        return ensemble_output, member_output
        


def _MLP_block(hidden_size):
    return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

def _init_layer(layer, generator):
    torch.nn.init.xavier_uniform_(layer.weight, generator=generator)
    layer.bias.data.fill_(0.01)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layer_num, generator):
        super(SimpleMLP, self).__init__()
        
        self.layers = nn.Sequential(
            
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[_MLP_block(hidden_size) for _ in range(hidden_layer_num - 1)], # unpack
            nn.Linear(hidden_size, output_size)
            
        )
        
        for layer in self.layers:
            if type(layer) == torch.nn.modules.linear.Linear:
                _init_layer(layer, generator)

    def forward(self, x):
        return self.layers(x)
    
class ImageWOOF_SimpleConvnet(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 20)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 35)

        _init_layer(self.conv1, generator)
        _init_layer(self.conv2, generator)

        self.fc1 = nn.Linear(4 * 18 * 18, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        _init_layer(self.fc1, generator)
        _init_layer(self.fc2, generator)
        _init_layer(self.fc3, generator)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

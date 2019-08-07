# original saved file with DataParallel
import torch

state_dict = torch.load('clf_ad1nl0_mri50_hippo30_lrflip_lenet_10.17.18.focal5.lr1e4.best.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
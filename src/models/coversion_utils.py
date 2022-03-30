from collections import OrderedDict
import json
import torch

# From https://stackoverflow.com/questions/64141188/how-to-load-checkpoints-across-different-versions-of-pytorch-1-3-1-and-1-6-x-u

def save_model_json(model, path):
    actual_dict = OrderedDict()
    for k, v in model.state_dict().items():
      actual_dict[k] = v.tolist()
    with open(path, 'w') as f:
      json.dump(actual_dict, f)

def load_model_json(model, path):
    data_dict = OrderedDict()
    with open(path, 'r') as f:
       data_dict = json.load(f)    
    own_state = model.state_dict()
    for k, v in data_dict.items():
        print('Loading parameter:', k)
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        if type(v) == list or type(v) == int:
            v = torch.tensor(v)
        own_state[k].copy_(v)
    model.load_state_dict(own_state)
    print('Model loaded')




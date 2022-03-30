# v1 ~ fake data from random torch arrays, loading real model
# python3 -m model_loader

import torch
import numpy as np
from model_functions import SimpleAE
from anomaly_functions import twin_ae_change_score
from collections import OrderedDict
import time

#from torch.profiler import profile, ProfilerActivity

def rename_state_dict_keys(state_dict):
    # Saved model contained differently named state_dict keys
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = str(key).replace("model.","")
        new_state_dict[new_key] = value
    return new_state_dict

def which_device(model):
    device = next(model.parameters()).device
    print("Model is on:", device)
    return device

def random_sample(batch_size, input_shape, device):
    return torch.randn(batch_size, *input_shape, dtype=torch.float).to(device)
    
    # Simulates conversion from a numpy array
    a=np.random.rand(batch_size, *input_shape)
    b=a.astype(np.float32)
    import pdb; pdb.set_trace()
    c=torch.tensor(b) # < fails without torch numpy support
    d=c.to(device)
 
    return d
    #return torch.tensor(np.random.rand(batch_size, *input_shape).astype(np.float32)).to(device)

def main():

    input_shape = (3, 32, 32)
    visualisation_channels = [0, 1, 2]
    model = SimpleAE(input_shape, visualisation_channels)
    print("Created model")
    from coversion_utils import save_model_json, load_model_json
    load_model_json(model, "model_ae.json")
    #state_dict = torch.load("model.pt") # ~ 13M sized file
    #state_dict = rename_state_dict_keys(state_dict)

    #model.load_state_dict(state_dict)
    model.eval()
    #model.eval().cuda() # < to put it all on GPU
    print("Loaded model:", model)

    device = which_device(model)

    # We have: model.forward .encode, .decode
    batch_size = 16
    # batch size 32 => 312 Mb in memory
    # batch size 64 => 625 Mb in memory
    number_of_batches = 50

    # warm-up
    #for i in range(3): twin_ae_change_score(model, random_sample(batch_size, input_shape, device), random_sample(batch_size, input_shape, device))

    time_total = 0
    if True:
        #with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        for attempt in range(number_of_batches):
            tensor_sample_1 = random_sample(batch_size, input_shape, device)
            tensor_sample_2 = random_sample(batch_size, input_shape, device)

            start_time = time.time()

            distance = twin_ae_change_score(model, tensor_sample_1, tensor_sample_2)
                
            if attempt == 0: print("Distance ", distance)

            end_time = time.time()
            time_total += (end_time - start_time)
            if attempt == 0: print("Single evaluation took ", time_total)

    print("Full evaluation took", time_total, "~ one batch in", time_total / number_of_batches)
    # ~ cpu_time_total cpu_memory_usage
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #prof.export_chrome_trace("trace.json") # open in: chrome://tracing

if __name__ == '__main__':
    main()


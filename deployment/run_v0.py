# v0 ~ fake data from random torch arrays, empty weights model
# python3 -m model_loader

import torch
import numpy as np
from model_functions import SimpleAE
from anomaly_functions import twin_ae_change_score
from collections import OrderedDict
import time
import math

#from torch.profiler import profile, ProfilerActivity

def which_device(model):
    device = next(model.parameters()).device
    print("Model is on:", device)
    return device

def random_sample(batch_size, input_shape, device):
    return torch.randn(batch_size, *input_shape, dtype=torch.float).to(device)
    
def main():

    input_shape = (3, 32, 32)
    visualisation_channels = [0, 1, 2]
    model = SimpleAE(input_shape, visualisation_channels)
    print("Created model (empty)")
    model.eval()

    device = which_device(model)

    # We have: model.forward .encode, .decode
    batch_size = 4
    # batch size 32 => 312 Mb in memory
    # batch size 64 => 625 Mb in memory
    number_of_batches = math.ceil(255. / float(batch_size))

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


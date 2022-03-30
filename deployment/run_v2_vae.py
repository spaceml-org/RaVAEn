# v2 ~ real data from loaded numpy arrays, loading real model
# python3 -m model_loader

import torch
import numpy as np
from model_functions import SimpleVAE
from anomaly_functions import twin_vae_change_score
from collections import OrderedDict
import time
import math
from image_utils import tiles2image, save_plot
from mem_report import mem_report
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
    latent_dim = 1024
    model = SimpleVAE(input_shape, latent_dim, visualisation_channels)
    print("Created model")

    from coversion_utils import save_model_json, load_model_json
    load_model_json(model, "model_vae.json")
    #state_dict = torch.load("model_vae.pt") # ~ 25M sized file
    #state_dict = rename_state_dict_keys(state_dict)

    #model.load_state_dict(state_dict)
    model.eval()
    #model.eval().cuda() # < to put it all on GPU
    print("Loaded model:", model)

    device = which_device(model)

    # We have: model.forward .encode, .decode
    batch_size = 8
    # batch size 16 => 312 Mb in memory
    # batch size 32 => 625 Mb in memory

    # warm-up
    #for i in range(3): twin_ae_change_score(model, random_sample(batch_size, input_shape, device), random_sample(batch_size, input_shape, device))

    # load data
    grid_shape = (17, 15)
    processed_inputs_1 = np.load("processed_inputs_1.npy", mmap_mode='r')
    processed_inputs_2 = np.load("processed_inputs_2.npy", mmap_mode='r')
    
    number_of_batches = math.ceil(len(processed_inputs_1) / batch_size)

    predicted_distances = np.zeros(len(processed_inputs_1))

    time_total = 0
    if True:
        #with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        for batch_n in range(number_of_batches):
            from_idx = int(batch_n*batch_size)
            to_idx = min(int((batch_n+1)*batch_size), len(processed_inputs_1))
            
            batch_1 = processed_inputs_1[from_idx : to_idx]
            batch_2 = processed_inputs_2[from_idx : to_idx]
            if batch_n == 0: print("example loaded data:", batch_1.shape, batch_2.shape, from_idx, "->", to_idx)

            tensor_sample_1 = torch.tensor(batch_1)
            tensor_sample_2 = torch.tensor(batch_2)

            start_time = time.time()

            distances = twin_vae_change_score(model, tensor_sample_1, tensor_sample_2)
            predicted_distances[from_idx : to_idx] = distances
            
            if batch_n == 0: print("distances ", distances.shape, distances)

            end_time = time.time()
            time_total += (end_time - start_time)
            if batch_n == 0: print("Single evaluation took ", time_total)

            if batch_n == 0: mem_report()
            del batch_1, batch_2, tensor_sample_1, tensor_sample_2 # clean from memory hopefully

    print("Full evaluation took", time_total, "~ one batch in", time_total / number_of_batches)
    # ~ cpu_time_total cpu_memory_usage
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #prof.export_chrome_trace("trace.json") # open in: chrome://tracing

    print(predicted_distances.shape)
    image_numpy = tiles2image(predicted_distances, grid_shape, overlap=0, tile_size = 32) # out as 1,w,h
    print("Saving image from tiles ~ ", image_numpy.shape, " as a plot.")
    try:
        print("Trying with matplotlib:")
        save_plot(image_numpy[0], "vae_prediction")

    except:
        print("... failed with matplotlib")

    try:
        print("Trying with pypng (no dependecies to matplotlib):")
        import png
        arr = image_numpy[0]
        normalised = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        png.from_array(normalised, 'L').save("vae_prediction_gray.png")
    except:
        print("... failed with pypng")
    
    print("--Finished!--")

if __name__ == '__main__':
    #import tracemalloc
    #tracemalloc.start()

    main()

    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics('lineno')

    #print("[ Top 10 memory eaters ]")
    #for stat in top_stats[:10]:
    #    print(stat)


from wandb_functions import get_data_from_id, get_runs_from_project, make_plot_from_ids
import numpy as np

entity = "mlpayloads"
project = "eval_paper_VAE_128small_DA_used" # 
#"eval_paper_VAE_128small_DA_used" #"eval_paper_VAE_128small_DA_not_used"
wanted_id = ""
runs = get_runs_from_project(entity, project, wanted_id)

print("PROJECT", project," has:")

for r in runs:
    print(r.name, r.id)



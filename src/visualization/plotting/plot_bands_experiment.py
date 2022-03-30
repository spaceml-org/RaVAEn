# Usage example:
# cd src/visualization/plotting/
# python plot_bands_experiment.py

# Adjust by changing the evaluation run IDs

#### PLAN:
"""
~ For one trained model we ideally have 4 evaluation runs on wandb.

evaluation_ids =
[
[["Floods from a1", "fires from a1", ...], ...]
[["Floods from b1", "fires from b1", ...],... ]
...
]

"a" can be 10 bands config, a1, a2, ... it's repetitions
"b" can be RGB only config, b1, b2, ... again it's repetitions

config_names =
["a", "b", ...]


Each configuration has one line on the graph with 4 points ~ floods, fires, ...
(Show both the avg point +- std marks)

Thus we can compare different configurations!
For sanity maybe only looking at the cos emb with memory 3.


"""

from wandb_functions import get_data_from_id, get_runs_from_project, make_plot_from_ids
import numpy as np


title = "Bands_experiment"
evaluation_ids = [
	[ # 10 bands # eval_paper_VAE_128small
		['1cpg7m0u', '498gpb84', '17ii8xkc', '3j5gvrsw'],
		['101pzflt', '3ouc57e0', '2mjis6dp', '896f31l9'],

	],
	[ # rgb+nir # eval_paper_VAE_128small_RGBNIR
		['2vey0186', '3eqzde1v', '3iwstssz', 'cmkiapti'],
		['2lj7p3li', 'ywrclaku', '2brkv033', '37wrldrg'],
	],
	[ # rgb # eval_paper_VAE_128small_RGB
		['3a1cd21u', '1p4c8klx', '2409pa1d', 'z0g1krgc'],
		['1udo0hej', '2ni30qhj', '18wrh7vn', '1greluup'],
	],
]

# ! keep the order :
dataset_names = ["floods", "fires", "hurricanes", "landscapes"]

# ! keep the order : 
config_names = ["10 bands", "rgb+nir", "rgb"]
wandb_project_names = ["eval_paper_VAE_128small", "eval_paper_VAE_128small_RGBNIR", "eval_paper_VAE_128small_RGB"]

# =============================================================================================

# Metric options:
"""
['cos_pixel | memory 1 | 32x32 - mean',
'diff_pixels | memory 1 | 32x32 - mean',
'cos_emb | memory 1 | 32x32 - mean',
'diff_emb | memory 1 | 32x32 - mean',
'KLDivEmbeddingImage | memory 1 | 32x32 - mean',
'WasserEmbeddingImage | memory 1 | 32x32 - mean',
'cos_pixel | memory 3 | 32x32 - mean', 
'cos_emb | memory 3 | 32x32 - mean']
"""


entity = "mlpayloads"

from pylab import plt
fig, axs = plt.subplots(2,2, figsize=(16,12)) # 1, 2 ~ two sideways
fig.suptitle(title)

used_metrics = ["cos_emb | memory 3 | 32x32 - mean", 
				"cos_pixel | memory 3 | 32x32 - mean",
				"cos_emb | memory 1 | 32x32 - mean", 
				"cos_pixel | memory 1 | 32x32 - mean"
				]
used_metrics_human_names = ["cos_emb_3", 
							"cos_pix_3",
							"cos_emb_1",
							"cos_pix_1"
							]

# 2x2 plot with COS embeddings and baselines, memory 1 and 3

metric = used_metrics[0]
metric_human_name = used_metrics_human_names[0]
make_plot_from_ids(axs[0, 0], entity, evaluation_ids, config_names, dataset_names, wandb_project_names, metric, metric_human_name)

metric = used_metrics[1]
metric_human_name = used_metrics_human_names[1]
make_plot_from_ids(axs[0, 1], entity, evaluation_ids, config_names, dataset_names, wandb_project_names, metric, metric_human_name)

metric = used_metrics[2]
metric_human_name = used_metrics_human_names[2]
make_plot_from_ids(axs[1, 0], entity, evaluation_ids, config_names, dataset_names, wandb_project_names, metric, metric_human_name)

metric = used_metrics[3]
metric_human_name = used_metrics_human_names[3]
make_plot_from_ids(axs[1, 1], entity, evaluation_ids, config_names, dataset_names, wandb_project_names, metric, metric_human_name)

plt.savefig("lastplot_"+title+".png")
#plt.show()

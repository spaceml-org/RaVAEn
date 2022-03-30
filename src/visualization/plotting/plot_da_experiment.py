# Usage example:
# cd src/visualization/plotting/
# python plot_*_experiment.py

# Adjust by changing the evaluation run IDs

from wandb_functions import get_data_from_id, get_runs_from_project, make_plot_from_ids
import numpy as np


"""
projects:

both bash files add to these:
eval_paper_VAE_128small_DA_not_used eval_paper_VAE_128small_DA_used


### PROJECT eval_paper_VAE_128small_DA_not_used  has:

# non DA models:
# ~ 1_NDAmodel_NDAdata

VAE_128small_1_NDAmodel_NDAdata_3k0vhd2o_epoch0_landslides 36ozfpk7
VAE_128small_1_NDAmodel_NDAdata_3k0vhd2o_epoch0_hurricanes 26i145dl
VAE_128small_1_NDAmodel_NDAdata_3k0vhd2o_epoch0_fires 15tlaug8
VAE_128small_1_NDAmodel_NDAdata_3k0vhd2o_epoch0_floods 2fh1lfqe


# ~ 3_DAmodel_NDAdata

VAE_128small_3_DAmodel_NDAdata_4fhse9aq_epoch0_landslides 2j5qyd6y
VAE_128small_3_DAmodel_NDAdata_4fhse9aq_epoch0_hurricanes wyuojp4m
VAE_128small_3_DAmodel_NDAdata_4fhse9aq_epoch0_fires 2t6vt212
VAE_128small_3_DAmodel_NDAdata_4fhse9aq_epoch0_floods 3bs2roy1


### PROJECT eval_paper_VAE_128small_DA_used  has:


# ~ 2_NDAmodel_DAdata

VAE_128small_2_NDAmodel_DAdata_3k0vhd2o_epoch0_landslides 31uge7kf
VAE_128small_2_NDAmodel_DAdata_3k0vhd2o_epoch0_hurricanes 2x3hy8w9
VAE_128small_2_NDAmodel_DAdata_3k0vhd2o_epoch0_fires 1b1zm2g1
VAE_128small_2_NDAmodel_DAdata_3k0vhd2o_epoch0_floods ry3vjmkr



# ~ 4_DAmodel_DAdata

VAE_128small_4_DAmodel_DAdata_4fhse9aq_epoch0_landslides 1acvtwnl
VAE_128small_4_DAmodel_DAdata_4fhse9aq_epoch0_hurricanes 2597zx3r
VAE_128small_4_DAmodel_DAdata_4fhse9aq_epoch0_fires 1b5k5wnu
VAE_128small_4_DAmodel_DAdata_4fhse9aq_epoch0_floods 28w5iocj




"""

title = "DA_experiment"
evaluation_ids = [
	[ # 1_NDAmodel_NDAdata # eval_paper_VAE_128small_DA_not_used
		['2fh1lfqe', '15tlaug8', '26i145dl', '36ozfpk7'],
	],
	[ # 2_NDAmodel_DAdata # eval_paper_VAE_128small_DA_used
		['ry3vjmkr', '1b1zm2g1', '2x3hy8w9', '31uge7kf'],
	],
	[ # 3_DAmodel_NDAdata # eval_paper_VAE_128small_DA_not_used
		['3bs2roy1', '2t6vt212', 'wyuojp4m', '2j5qyd6y'],
	],
	[ # 4_DAmodel_DAdata # eval_paper_VAE_128small_DA_used
		['28w5iocj', '1b5k5wnu', '2597zx3r', '1acvtwnl'],
	],
]

# ! keep the order :
dataset_names = ["floods", "fires", "hurricanes", "landscapes"]

# ! keep the order : 
config_names = ["1_NDAmodel_NDAdata", "2_NDAmodel_DAdata", "3_DAmodel_NDAdata", "4_DAmodel_DAdata"]
wandb_project_names = ["eval_paper_VAE_128small_DA_not_used", "eval_paper_VAE_128small_DA_used", "eval_paper_VAE_128small_DA_not_used", "eval_paper_VAE_128small_DA_used"]

# =============================================================================================

# Metric options:
"""
['cos_pixel | memory 1 | 30x30 - mean',
'diff_pixels | memory 1 | 30x30 - mean',
'cos_emb | memory 1 | 30x30 - mean',
'diff_emb | memory 1 | 30x30 - mean',
'KLDivEmbeddingImage | memory 1 | 30x30 - mean',
'WasserEmbeddingImage | memory 1 | 30x30 - mean',
'cos_pixel | memory 3 | 30x30 - mean', 
'cos_emb | memory 3 | 30x30 - mean']
"""


entity = "mlpayloads"

from pylab import plt
fig, axs = plt.subplots(2,2, figsize=(16,12)) # 1, 2 ~ two sideways
fig.suptitle(title)

used_metrics = ["cos_emb | memory 3 | 30x30 - mean", 
				"cos_pixel | memory 3 | 30x30 - mean",
				"cos_emb | memory 1 | 30x30 - mean", 
				"cos_pixel | memory 1 | 30x30 - mean"
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
plt.show()


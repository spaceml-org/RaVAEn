import pandas as pd
import wandb
import json
from os import listdir
from os.path import isfile, join
import urllib.request
import numpy as np

api = wandb.Api()
#mlpayloads/eval_paper_VAE_100


def load_table(run, desired_table_name):
    # Omg... also pretty slow
    desired_file = [p for p in run.files() if desired_table_name in p.name][0]
    data = ""
    for line in urllib.request.urlopen(desired_file.direct_url):
        data += line.decode('utf-8')
    return json.loads(data)


def get_runs_from_project(entity, project, filter_name = ''):
    runs = api.runs(entity + "/" + project)
    #print("We got", len(runs), "total runs")
    
    if len(filter_name) > 0:
        all_names = [r.name for r in runs]
        runs = [r for r in runs if filter_name in r.name]
    return runs

def get_data_from_id(runs, wanted_id, desired_method_name = "cos_emb | memory 3 | 32x32 - mean"):

    ids = [r.id for r in runs]
    names = [r.name for r in runs]
    desired_table_name = "Detection technique summary statistics"

    if wanted_id in ids:
        for i, run in enumerate(runs):
            if wanted_id == ids[i]:
                # We have the wanted run
                json_data = load_table(run, desired_table_name)
                # json_data['columns'] ~ ['Detection method', 'area under precision recall curve', 'precision at 100% recall']

                #print(json_data['data'])
                metric_names = [json_data['data'][i][0] for i in range(len(json_data['data']))]
                try:
                    method_idx = metric_names.index(desired_method_name)
                except:
                    print("didn't find desired metric (",desired_method_name,") in existing metrics:", metric_names)
                    method_idx = metric_names.index(desired_method_name)

                metric_values = json_data['data'][method_idx] # ['cos_emb | memory 3 | 32x32 - mean', 0.27483912715449255, 0.06895697178660265]
                metric_value = metric_values[1] # we want the 'area under precision recall curve'
                
                #print(desired_method_name, "=", metric_value)
                return metric_value

    else:
        print("Couldn't find desired id", wanted_id, "among", ids)




def make_plot_from_ids(ax, entity, evaluation_ids, config_names, dataset_names, wandb_project_names,
                        metric, metric_human_name):
    experiments_data = []

    ax.title.set_text(metric_human_name)


    for config_i, ids_of_config in enumerate(evaluation_ids):
        project = wandb_project_names[config_i]
        runs = get_runs_from_project(entity, project) # shared for all in this project

        config_data = {} # prepared to add all 'floods', ... together
        for d in dataset_names:
            config_data[d] = []

        for repeat_i, wanted_ids in enumerate(ids_of_config):
            for dataset_i,wanted_id in enumerate(wanted_ids):
                dataset_name = dataset_names[dataset_i]
                value = get_data_from_id(runs, wanted_id, metric)

                print("config '"+config_names[config_i]+"'","#"+str(repeat_i), dataset_name,"has",value)
                config_data[dataset_name].append(value)


        print(config_names[config_i], "with", metric_human_name, "=>", config_data)
        experiments_data.append(config_data)

    ### Plot:

    """
    3 lines:
    with 4 points
    avg from two vals each
    {'floods': [0.4604094197656117, 0.4372921872918173], 'fires': [0.9089454771249286, 0.9055039053687104], 'hurricanes': [0.7469784648548236, 0.7502675607970039], 'landscapes': [0.7332296393478988, 0.7606667375781838]}

    {'floods': [0.27483912715449255, 0.2908902184658966], 'fires': [0.8240672136093059, 0.8155723152254265], 'hurricanes': [0.4824828809308839, 0.4704292336585485], 'landscapes': [0.6997034003848734, 0.7074805476985974]}

    {'floods': [0.3167572973687815, 0.30007053059184774], 'fires': [0.8690815616082604, 0.880484739083317], 'hurricanes': [0.7418186578340848, 0.7647391269632122], 'landscapes': [0.7547858660808782, 0.7374434652491647]}

    """

    #import pdb; pdb.set_trace()

    for config_i, config_data in enumerate(experiments_data):
        points = []
        stds = []
        for dataset_i, dataset_name in enumerate(config_data):
            dataset_data = config_data[dataset_name]

            avg_value = np.mean(dataset_data)
            std_value = np.std(dataset_data)
            points.append(avg_value)
            stds.append(std_value)
            
        x = list(range(len(points)))
        y = points
        e = stds
        print(x, y, e)

        #ax.plot(points, 'o-', label=config_names[config_i])
        ax.errorbar(x, y, e, linestyle='-', marker='o', label=config_names[config_i]+", "+metric_human_name)

    ax.set_ylim([0.0,1.0])
    ax.set_xticks(np.arange(len(dataset_names))) 
    ax.set_xticklabels(dataset_names)

    ax.legend()

    ax.set_ylabel("area under precision recall curve")
    return ax



if False:
    entity, project = "mlpayloads", "eval_paper_VAE_128small_RGB"  # set to your entity and project#
    runs = get_runs_from_project(entity, project)

    value = get_data_from_id(runs, wanted_id = "3a1cd21u")
    print("!")

    assert False


    summary_list, config_list, name_list = [], [], []
    for idx, run in enumerate(runs):
        summary = run.summary._json_dict
        if desired_table_name not in summary.keys():
            continue

        json_data = load_table(run, desired_table_name)

        print(json_data)
        print("----")
        df = pd.DataFrame(data=json_data)

        print(df)


        summary_list.append(summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)


    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })

    runs_df.to_csv("projects.csv")

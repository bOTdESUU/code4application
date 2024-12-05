# %%
from plotting.read_results import *

# %%
EXP_NAME = 'lfq_K2pow16_zdim16_wt_tf'
n_seed = 5
datasets = ['antmaze-large-diverse-v0',]

# %%
args = Parser().parse_args()
args.exp_name = EXP_NAME
args.dataset = [dataset]

# for dataset in ([args.dataset] if args.dataset else DATASETS):
for dataset in datasets:
    subdirs = glob.glob(os.path.join(LOGBASE, dataset))

    for subdir in subdirs:
        reldir = subdir.split('/')[-1]
        paths = glob.glob(os.path.join(subdir, args.exp_name+"*", TRIAL))

        mean, err, scores, infos = load_results(paths)
        print(f'{dataset.ljust(30)} | {len(scores)} scores | score {mean:.2f} +/- {err:.2f} | '
                f'return {infos["returns"]:.2f} | first value {infos["first_value"]:.2f} | '
                f'first_search_value {infos["first_search_value"]:.2f} | step: {infos["step"]:.2f} | '
                f'prediction_error {infos["prediction_error"]:.2f} | discount_return {infos["discount_return"]:.2f}'
                )

# %%
import glob
import os

def load_result(path):
	'''
		path : path to experiment directory; expects `rollout.json` to be in directory
	'''
	#path = os.path.join(path, "0")
	# fullpath = os.path.join(path, 'rollout.json')
	fullpath = path
	suffix = path.split('/')[-1]

	if not os.path.exists(fullpath):
		return None, None

	results = json.load(open(fullpath, 'rb'))
	score = results['score']
	info = dict(returns=results["return"],
				first_value=results["first_value"],
				first_search_value=results["first_search_value"],
                discount_return=results["discount_return"],
				prediction_error=results["prediction_error"],
				step=results["step"])

	return score * 100, info
# Step 1: Find all rollout.json files
rollout_files = glob.glob('logs/*/*/*/rollout.json')

import pandas as pd
df = pd.DataFrame(columns=['Dataset', 'ExpName', 'Seed', 'Rollout Time', 'Scores', 'returns', 'first_value', 'first_search_value', 'step', 'prediction_error', 'discount_return'])
rows = []

# Step 2: Iterate over each file path
for file_path in rollout_files:
    # Split the path into parts
    parts = file_path.split('/')
    
    # Step 3: Extract parts of the path
    dataset = parts[1]
    exp_name = parts[2].rsplit('-', 1)[0]  # Remove the seed part
    seed = parts[2].split('-')[-1]  # Get the seed part
    rollout_time = int(parts[3])
    
    scores, infos = load_result(file_path)
    # Step 4: Print or store the parsed information
    # print(f'Dataset: {dataset}, ExpName: {exp_name}, Seed: {seed}, Rollout Time: {rollout_time}')
    # print(f'Scores: {scores}, Infos: {infos}')
    
    new_row = {'dataset': dataset, 'exp_name': exp_name, 'seed': seed, 'rollout_time': rollout_time, 'scores': scores, **infos}
    rows.append(new_row)
    
    # # Create a new row for the result in the dataframe
    # result_df = result_df.concat({'Dataset': dataset, 'Scores': scores, 'Mean': mean, 'Error': err, 'Returns': infos['returns'], 'First Value': infos['first_value'], 'First Search Value': infos['first_search_value'], 'Step': infos['step'], 'Prediction Error': infos['prediction_error'], 'Discount Return': infos['discount_return']}, ignore_index=True)

df = pd.DataFrame(rows, index=None)

# %%
df = df[df['exp_name'] != 'debug']
df = df[~df['exp_name'].str.contains('test')]
df = df.sort_values(by=['dataset','exp_name', 'seed', 'rollout_time'])
df.reset_index(drop=True, inplace=True)
df

# %%
df.to_csv('dataframe.csv', index=False)

# %%
import pandas as pd

# Load the dataframe
df_loaded = pd.read_csv('dataframe.csv')

# Assert the contents of the loaded dataframe
assert df_loaded.equals(df)

# %%
df_loaded

# %%




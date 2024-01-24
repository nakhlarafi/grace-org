import json
import numpy as np
import sys

project_name = sys.argv[1]

def calculate_metrics(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    top1, top3, top5, top10 = 0, 0, 0, 0
    mfr_list, mar_list = [], []

    for project, info in data.items():
        ranking = info['ranking']
        ground_truth = info['ans']

        min_rank, ranks = float('inf'), []

        for gt in ground_truth:
            rank = ranking.index(gt)
            ranks.append(rank)
            min_rank = min(min_rank, rank)

        mfr_list.append(min_rank)
        mar_list.append(np.mean(ranks))

        # Update top-k counts based on minimum rank
        if min_rank == 0:
            top1 += 1
        if min_rank < 3:
            top3 += 1
        if min_rank < 5:
            top5 += 1
        if min_rank < 10:
            top10 += 1

    num_projects = len(data)
    mfr_avg = np.mean(mfr_list)
    mar_avg = np.mean(mar_list)

    return {
        'Top-1': top1,
        'Top-3': top3,
        'Top-5': top5,
        'Top-10': top10,
        'MFR': mfr_avg,
        'MAR': mar_avg
    }

# Usage example
json_file = f'crossvalidation/{project_name}/{project_name}_merged_data.json'
metrics = calculate_metrics(json_file)
print(metrics)

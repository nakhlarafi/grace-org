import json
import numpy as np
import sys

project_name = sys.argv[1]
def calculate_metrics(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    top_count = [0] * 5  # Initialize top count for positions 1 to 5
    mfr_list, mar_list = [], []

    for project, info in data.items():
        ranking = info['ranking']
        ground_truth = info['ans']

        ranks, min_rank = [], float('inf')

        for gt in ground_truth:
            rank = ranking.index(gt)
            ranks.append(rank)
            min_rank = min(min_rank, rank)

            # Update top count for this rank
            if rank < len(top_count):
                top_count[rank] += 1

        mfr_list.append(min_rank)
        mar_list.append(np.mean(ranks))

    top1 = top_count[0]
    top3 = sum(top_count[:3])
    top5 = sum(top_count[:5])
    mfr_avg = np.mean(mfr_list)
    mar_avg = np.mean(mar_list)

    return {
        'Top-1': top1,
        'Top-3': top3,
        'Top-5': top5,
        'MFR': mfr_avg,
        'MAR': mar_avg
    }

# Usage example
json_file = f'crossvalidation/{project_name}/{project_name}_merged_data.json'  # Replace with your JSON file path
metrics = calculate_metrics(json_file)
print(metrics)

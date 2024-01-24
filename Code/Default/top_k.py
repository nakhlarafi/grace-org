import json
import numpy as np

def calculate_metrics(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    top1, top3, top5, top10 = 0, 0, 0, 0
    mfr_list, mar_list = []

    for project, info in data.items():
        ranking = info['ranking']
        ground_truth = info['ans']

        min_rank, ranks = float('inf'), []

        for gt in ground_truth:
            rank = ranking.index(gt)
            ranks.append(rank)
            min_rank = min(min_rank, rank)

        # Update counts for top-n metrics
        if min_rank == 0:
            top1 += 1
        if min_rank < 3:
            top3 += 1
        if min_rank < 5:
            top5 += 1
        if min_rank < 10:
            top10 += 1

        mfr_list.append(min_rank)
        mar_list.append(np.mean(ranks))

    num_projects = len(data)
    mfr_avg = np.mean(mfr_list)
    mar_avg = np.mean(mar_list)

    return {
        'Top-1': top1 / num_projects,
        'Top-3': top3 / num_projects,
        'Top-5': top5 / num_projects,
        'Top-10': top10 / num_projects,
        'MFR': mfr_avg,
        'MAR': mar_avg
    }

# # Usage example
# json_file = 'path_to_your_json_file.json'  # Replace with your JSON file path
# metrics = calculate_metrics(json_file)
# print(metrics)

# Usage example
json_file = f'crossvalidation/{project_name}/{project_name}_merged_data.json'  # Replace with your JSON file path
metrics = calculate_metrics(json_file)
print(metrics)

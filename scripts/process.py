from collections import defaultdict
import numpy as np


def average_metric_arrays(dates, nested_arrays)-> list:
    date_to_arrays = defaultdict(list)

    for date, array in zip(dates, nested_arrays):
        date_to_arrays[date].append(array)

    averaged_results = []
    for date, arrays in date_to_arrays.items():
        averaged_array = [sum(x) / len(x) for x in zip(*arrays)]
        averaged_results.append(averaged_array)

    return averaged_results

def parse_dates_metrics(dates_file, metrics_file)-> tuple:
    dates =  np.load(dates_file)["arr_0"]
    dates = np.array([np.datetime64(date) for date in dates])
    metrics = np.load(metrics_file)["arr_0"]
    metrics = np.array(average_metric_arrays(dates, metrics))
    return dates, metrics
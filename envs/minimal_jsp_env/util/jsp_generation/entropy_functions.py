from scipy.stats import entropy
from collections import Counter
from typing import List


def calculate_entropy_from_operations_list(operation_types: List) -> float:
    counts = Counter(operation_types)
    probabilities = [count/len(operation_types) for count in counts.values()]
    entropy_val = entropy(probabilities)
    return entropy_val

def collect_all_operations(jobs: List) -> List[tuple]:
    all_operations = []
    for job in jobs:
        job_operations = [(i.machine_type, i.duration) for i in job]
        all_operations += job_operations
    return all_operations
# Author: Reef Lakin
# Last Modified: 30.04.2025
# Description: A file to test the dataset quality and ensure that the data is loaded correctly.

# ========== IMPORTS ==========
from utilities.preprocessing import load_dataset_for_stgcn, split_dataset


# ========== TEST Lines ==========
dataset = load_dataset_for_stgcn(window_size=12)

train, test, val = split_dataset(dataset)
print("Training set size:", train.snapshot_count)
print("Testing set size:", test.snapshot_count)
print("Validation set size:", val.snapshot_count)

first_item = next(iter(dataset))
print(type(first_item))
print(first_item)

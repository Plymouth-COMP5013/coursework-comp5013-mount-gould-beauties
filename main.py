# Imports
from utilities.preprocessing import load_dataset_for_stgcn, split_dataset
from utilities.display import get_target_velocities

dataset = load_dataset_for_stgcn(window_size=12)

train, test, val = split_dataset(dataset)

print("Training set size:", train.snapshot_count)
print("Testing set size:", test.snapshot_count)
print("Validation set size:", val.snapshot_count)
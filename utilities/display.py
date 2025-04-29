# Author: Reef Lakin
# Last Modified: 28.04.2025
# Description: Contains functions for displaying STGCN data in a very human-friendly way.

from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def get_target_velocity_for_station(dataset: StaticGraphTemporalSignal, time_index, node_index):
	"""
	Get the target velocity for a specific time index and node index.

	Args:
		dataset (StaticGraphTemporalSignal): The dataset in a StaticGraphTemporalSignal format.
		time_index (int): The start time index for which to get the target velocity.
		node_index (int): The node index (i.e., station) for which to get the target velocity.

	Returns:
		target: The target velocity for the specified node at the specified time index.
	"""
	
	target = dataset._get_target(time_index)[node_index]
	return target


def get_target_velocities(dataset: StaticGraphTemporalSignal, time_index):
	"""
	Get the target velocities for all stations at a specific time index.

	Args:
		dataset (StaticGraphTemporalSignal): The dataset in a StaticGraphTemporalSignal format.
		time_index (int): The start time index for which to get the target velocities.

	Returns:
		targets: The target velocities for all stations at the specified time index.
	"""
	
	targets = dataset._get_target(time_index)
	return targets
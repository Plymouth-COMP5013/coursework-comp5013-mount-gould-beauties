import torch

class ZScoreNormaliser:

	def __init__(self, values: torch.Tensor):
		self.mean = values.mean()
		self.std = values.std()

	def normalise(self, values: torch.Tensor) -> torch.Tensor:
		"""
		Normalise the input values using Z-score normalisation.

		Args:
			values (torch.Tensor): The input values to be normalised.

		Returns:
			torch.Tensor: The normalised values.
		"""
		return (values - self.mean) / self.std
	
	def denormalise(self, values: torch.Tensor) -> torch.Tensor:
		"""
		Denormalise the input values using Z-score normalisation.

		Args:
			values (torch.Tensor): The input values to be denormalised.

		Returns:
			torch.Tensor: The denormalised values.
		"""
		return (values * self.std) + self.mean
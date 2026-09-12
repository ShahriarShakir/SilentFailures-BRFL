
from typing import Any, Dict, List

class FLServer:
	def __init__(self, num_clients: int, strategy: str = 'fedavg'):
		self.num_clients = num_clients
		self.strategy = strategy
		self.global_model = None
		self.client_models = [None] * num_clients

	def aggregate(self, client_weights: List[Any]) -> Any:
		if self.strategy == 'fedavg':
			return self.fedavg(client_weights)
		elif self.strategy == 'fedbn':
			return self.fedbn(client_weights)
		else:
			raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

	def fedavg(self, client_weights: List[Any]) -> Any:
		# Simple FedAvg: average weights (assume list of state_dicts)
		avg_weights = {}
		for key in client_weights[0].keys():
			avg_weights[key] = sum([cw[key] for cw in client_weights]) / len(client_weights)
		return avg_weights

	def fedbn(self, client_weights: List[Any]) -> Any:
		# FedBN: average all except BN layers (keys with 'bn' in name)
		avg_weights = {}
		for key in client_weights[0].keys():
			if 'bn' in key:
				avg_weights[key] = client_weights[0][key]  # do not average BN
			else:
				avg_weights[key] = sum([cw[key] for cw in client_weights]) / len(client_weights)
		return avg_weights

class FLClient:
	def __init__(self, client_id: int, model: Any, data: Any):
		self.client_id = client_id
		self.model = model
		self.data = data

	def train(self, epochs: int = 1):
		# Example: call model's train method (to be replaced with actual trainer)
		# self.model.train(data=self.data, epochs=epochs)
		pass

	def get_weights(self) -> Any:
		# Return model weights (state_dict)
		return self.model.state_dict()

	def set_weights(self, weights: Any):
		# Set model weights (state_dict)
		self.model.load_state_dict(weights)


def get_model_parameters(model: Any) -> Any:
	return model.state_dict()


def set_model_parameters(model: Any, params: Any):
	model.load_state_dict(params)

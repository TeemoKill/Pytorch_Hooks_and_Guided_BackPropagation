import os

import numpy as np
import torch


class ResultStatistics():

	def __init__(self, dir):
		self.dir = dir

		self.layer1_norms_stack = None
		self.layer2_norms_stack = None

		self._read_norm_tensors()

	def _read_norm_tensors(self):
		layer1_files = []
		layer2_files = []
		for root, dirs, files in os.walk(self.dir):
			for file in files:
				if file[-(len(".pt")+1)] == "1":
					layer1_files.append(file)
				elif file[-(len(".pt")+1)] == "2":
					layer2_files.append(file)

		def read_file_list(file_list, out_tensor=None):
			for i, name in enumerate(file_list):
				file_path = os.path.join(self.dir, name)
				norm_tensor = torch.load(file_path)

				if out_tensor is None:
					out_tensor = norm_tensor
				else:
					out_tensor = torch.cat(
						[out_tensor, norm_tensor],
						dim=0
					)
			return out_tensor

		self.layer1_norms_stack = read_file_list(layer1_files)
		self.layer2_norms_stack = read_file_list(layer2_files)

	def layer1_percentile(self, p):
		return self._percentile(self.layer1_norms_stack, p)

	def layer2_percentile(self, p):
		return self._percentile(self.layer2_norms_stack, p)

	def _percentile(self, t, p):
		def percentile_1d(t_1d, p):
			# Thanks to https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
			t_np = t_1d.numpy()
			result = np.percentile(t_np, p)
			result = torch.tensor(result).unsqueeze(0)
			return result

		result = []
		for tensor in t.transpose(0, 1):
			channelwise_percentile = percentile_1d(tensor, p)
			result.append(channelwise_percentile)

		result = torch.cat(result, dim=0)
		return result

	def layer1_median(self):
		return self._median(self.layer1_norms_stack)

	def layer2_median(self):
		return self._median(self.layer2_norms_stack)

	def _median(self, t):
		return torch.median(t, dim=0).values


if __name__ == '__main__':
	VGG16_statistics = ResultStatistics("VGG16_L2Norm")
	VGG16bn_statistics = ResultStatistics("VGG16bn_L2Norm")

	VGG16_percentiles = {}
	VGG16bn_percentiles = {}

	VGG16_medians = {}
	VGG16bn_medians = {}

	VGG16_medians["layer1"] = VGG16_statistics.layer1_median()
	VGG16_medians["layer2"] = VGG16_statistics.layer2_median()

	VGG16bn_medians["layer1"] = VGG16bn_statistics.layer1_median()
	VGG16bn_medians["layer2"] = VGG16bn_statistics.layer2_median()


	for p in range(5, 100, 5):
		VGG16_percentiles[p] = {}
		VGG16_percentiles[p]["layer1"] = VGG16_statistics.layer1_percentile(p)
		VGG16_percentiles[p]["layer2"] = VGG16_statistics.layer2_percentile(p)

		VGG16bn_percentiles[p] = {}
		VGG16bn_percentiles[p]["layer1"] = VGG16bn_statistics.layer1_percentile(p)
		VGG16bn_percentiles[p]["layer2"] = VGG16bn_statistics.layer2_percentile(p)

	print("Done")

import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms

from Dataloader import get_Dataloader

import matplotlib.pyplot as plt


# def var_hook(grad):
# 	print(grad)


class Hooker():

	def __init__(self, model, output_path):
		self.model = model
		self.output_path = output_path
		# self.forward_feat_in = []
		# self.forward_feat_out = []
		# self.backward_feat_in = []
		# self.backward_feat_out = []
		self._current_file_name = None

		self._layer1_norm = None
		self._layer1_grad = None

		self.model.eval()
		self.register_hooks()

	def register_hooks(self):
		def module_forward_hook(module, input, output):
			# This hook is not used in the homework submission
			# I just put it here

			# print(module)
			# print("input: ", input)
			# print("output: ", output)
			print("Im just a forward module hook doing nothing")

			# self.forward_feat_in.append(input)
			# self.forward_feat_out.append(output)

		def module_backward_hook1(module, grad_input, grad_output):
			# print("Module_Name: ", module)
			# print("grad_output: ", grad_output)
			# print("grad_input: ", grad_input)

			if isinstance(module, nn.Conv2d):
				grad_input_tensor = grad_input[0]
				l2_norm = self.channelwise_l2_norm(grad_input_tensor)

				self._layer1_grad = grad_input_tensor
				self._layer1_norm = l2_norm

				# torch.save(tensor, file_path)
				self.save_norm_to_file(l2_norm, 1)

			# self.backward_feat_in.append(grad_input)
			# self.backward_feat_out.append(grad_output)

		def module_backward_hook2(module, grad_input, grad_output):
			# print("Module_Name: ", module)
			# print("grad_output: ", grad_output)
			# print("grad_input: ", grad_input)

			if isinstance(module, nn.Conv2d):
				grad_input_tensor = grad_input[0]
				l2_norm = self.channelwise_l2_norm(grad_input_tensor)

				self.save_norm_to_file(l2_norm, 2)

		modules = list(self.model.named_children())
		name, features_module = modules[0]
		sub_modules = list(features_module.named_children())

		first_layer = True
		for i, (sub_name, sub_module) in enumerate(sub_modules):
			if i < 4:
				if isinstance(sub_module, nn.Conv2d):
					if first_layer:
						sub_module.register_backward_hook(module_backward_hook1)
						first_layer = False
					else:
						sub_module.register_backward_hook(module_backward_hook2)
					print(f"{sub_module} -> backward_hook registered")
			else:
				break
		#
		# for name, module in modules:
		# 	for sub_name, sub_module in module.named_children():
		# 		# sub_module.register_forward_hook(module_forward_hook)
		# 		sub_module.register_backward_hook(module_backward_hook)

	def channelwise_l2_norm(self, t):
		norm = torch.norm(t, p=2, dim=(2, 3))
		return norm

	def save_norm_to_file(self, norm_tensor, layer):
		save_name = f"{self._current_file_name}_{layer}.pt"
		file_path = os.path.join(self.output_path, save_name)

		torch.save(norm_tensor, file_path)


	def process(self, input_dict):
		image = input_dict["image"]
		file_name = input_dict["filename"][0]
		self._current_file_name = os.path.basename(file_name)

		image = image.requires_grad_(False)
		image = image.requires_grad_()
		# set requires_grad_ to False then back to cancel out previous gradient
		model_output = self.model(image)
		self.model.zero_grad()
		pred_class = model_output.argmax().item()

		grad_target_map = torch.zeros(
			model_output.shape,
			dtype=torch.float
		)
		grad_target_map[0][pred_class] = 1

		model_output.backward(grad_target_map)

		# result = self.backward_feat_in[-1][0].data[0]  # .permute(1, 2, 0)
		# print(result.shape)

		return self._layer1_grad, self._layer1_norm





if __name__ == '__main__':
	from guidedbpcodehelpers import imshow2

	VGG16 = models.vgg16(pretrained=True)
	VGG16_bn = models.vgg16_bn(pretrained=True)

	hooker_VGG16 = Hooker(VGG16, "VGG16_L2Norm")
	hooker_VGG16bn = Hooker(VGG16_bn, "VGG16bn_L2Norm")
	dataloader = get_Dataloader()

	for i in range(250):
		inputs = next(iter(dataloader))

		for hooker in [hooker_VGG16, hooker_VGG16bn]:
			layer1_grad, layer1_norm = hooker.process(inputs)

			# image = inputs["image"]
			# imshow2(layer1_grad, image)
			# imshow2(result, image)

		if i%10 == 0:
			print(f"image {i} processed")

#
# total_feat_out = []
# total_feat_in = []
#
# def module_forward_hook(module, input, output):
# 	print(module)
# 	print("input: ", input)
# 	print("output: ", output)
# 	total_feat_out.append(output)
# 	total_feat_in.append(input)
#
#
# def module_backward_hook(module, grad_input, grad_output):
# 	print("Module_Name: ", module)
# 	print("grad_output: ", grad_output)
# 	print("grad_input: ", grad_input)
#
# 	if isinstance(module, nn.Conv2d):
# 		grad_input_tensor = grad_input[0]
# 		l2_norm = grad_input_tensor.norm(2, 1, True)
# 		norm = F.normalize(grad_input_tensor, p=2, dim=1)
#
# 	total_feat_in.append(grad_input)
# 	total_feat_out.append(grad_output)
#
#
#
# modules = VGG16.named_children()
# for name, module in modules:
# 	# module.register_forward_hook(module_forward_hook)
# 	for sub_name, sub_module in module.named_children():
# 		sub_module.register_backward_hook(module_backward_hook)
#
#
# def normalize(I):
# 	# 归一化梯度map，先归一化到 mean=0 std=1
# 	norm = (I - I.mean()) / I.std()
# 	# 把 std 重置为 0.1，让梯度map中的数值尽可能接近 0
# 	norm = norm * 0.1
# 	# 均值加 0.5，保证大部分的梯度值为正
# 	norm = norm + 0.5
# 	# 把 0，1 以外的梯度值分别设置为 0 和 1
# 	norm = norm.clip(0, 1)
# 	return norm
#
#
# dataloader = get_Dataloader()
# inputs = next(iter(dataloader))
#
# image = inputs["image"]
# print(image.shape)
# file_name = inputs["filename"]
#
# image = image.requires_grad_()
#
# model_output = VGG16(image)
# VGG16.zero_grad()
# pred_class = model_output.argmax().item()
#
# grad_target_map = torch.zeros(
# 	model_output.shape,
# 	dtype=torch.float
# )
# grad_target_map[0][pred_class] = 1
#
# model_output.backward(grad_target_map)
# print(type(total_feat_in[-1]))
# print(type(total_feat_in[-1][0]))
#
# result = total_feat_in[-1][0].data[0]  # .permute(1, 2, 0)
# print(result.shape)
# plt.imshow(normalize(result.permute(1, 2, 0).numpy()))
# plt.show()
#
# imshow2(result, image)
#
#
# print("========== saved inputs and outputs ==========")
# for idx in range(len(total_feat_in)):
# 	print("input: ", total_feat_in[idx])
# 	print("output: ", total_feat_out[idx])






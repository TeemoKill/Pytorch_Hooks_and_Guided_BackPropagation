import torch
from torch import nn

from guidedbpcodehelpers import setbyname


class CustomReLU(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		grad_input[grad_output < 0] = 0
		return grad_input


class CustomReluModule(nn.Module):

	def __init__(self):
		super(CustomReluModule, self).__init__()

	def forward(self, input):
		return CustomReLU.apply(input)

	def extra_repr(self):
		repr_str = "Hello here is the Custom ReLU Module"
		return repr_str


class Guided_backprop():
	def __init__(self, model):
		self.model = model
		self.image_reconstruction = None
		self.activation_maps = []
		self.model.eval()
		# self.register_hooks()
		self.replace_modules()

	def replace_modules(self):
		def recursively_replace_modules(module, inherit_name=None):
			for sub_name, sub_module in module.named_children():
				target_name = []
				if inherit_name is not None:
					target_name.append(inherit_name)
				target_name.append(sub_name)
				target_name = ".".join(target_name)
				if isinstance(sub_module, nn.Sequential):
					recursively_replace_modules(sub_module, target_name)
				elif isinstance(sub_module, nn.ReLU):
					success = setbyname(self.model, target_name, CustomReluModule())
					print(f"replace {target_name} with CustomReLU {'success' if success else 'fail'}")

		recursively_replace_modules(self.model, None)

		# setbyname(self.model, "features.1", CustomReluModule())

	def register_hooks(self):
		def first_layer_hook_fn(module, grad_in, grad_out):
			self.image_reconstruction = grad_in[0]

		def forward_hook_fn(module, input, output):
			self.activation_maps.append(output)

		def backward_hook_fn(module, grad_in, grad_out):
			grad = self.activation_maps.pop()

			grad[grad > 0] = 1

			positive_grad_out = torch.clamp(grad_out[0], min=0.0)

			new_grad_in = positive_grad_out * grad

			return (new_grad_in,)

		modules = list(self.model.features.named_children())
		# modules = list(self.model.named_children())  # for VGG model
		print(len(modules))

		for name, module in modules:
			print(f"{name}")
			if isinstance(module, nn.ReLU):
				print(f"^^ registering hooks")
				module.register_forward_hook(forward_hook_fn)
				module.register_backward_hook(backward_hook_fn)

		first_layer = modules[0][1]
		first_layer.register_backward_hook(first_layer_hook_fn)

	def process(self, input_image, target_class):
		input_image.requires_grad_()
		model_output = self.model(input_image)
		self.model.zero_grad()
		pred_class = model_output.argmax().item()

		grad_target_map = torch.zeros(
			model_output.shape,
			dtype=torch.float
		)

		if target_class is not None:
			grad_target_map[0][target_class] = 1
		else:
			grad_target_map[0][pred_class] = 1

		model_output.backward(grad_target_map)

		# result = self.image_reconstruction.data[0].permute(1, 2, 0)
		result = input_image.grad.data
		print(result.shape)
		return result


if __name__ == '__main__':
	from torchvision import models, transforms

	from Dataloader import get_Dataloader
	from guidedbpcodehelpers import imshow2

	dataloader = get_Dataloader()

	model = models.vgg16(pretrained=True)
	guided_bp = Guided_backprop(model)

	show_how_many_images = 1
	for i in range(show_how_many_images):
		inputs = next(iter(dataloader))

		image = inputs["image"]
		print(image.shape)
		file_name = inputs["filename"]

		result = guided_bp.process(image, None)

		imshow2(result, image)

	print('END')

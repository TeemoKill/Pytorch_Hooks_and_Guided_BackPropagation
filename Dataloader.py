from torch.utils.data import DataLoader
from torchvision import transforms

from imgnetdatastuff import dataset_imagenetvalpart


def get_Dataloader():
	data_dir = "./mnt/scratch1/data/imagespart/"
	xmllabeldir = "./xmllabel/"
	synsetfile = "./synset_words.txt"
	maxnum = -1

	means = [0.485, 0.456, 0.406]
	stds = [0.229, 0.224, 0.225]
	size = 224

	transform = transforms.Compose([
		transforms.Resize(size),
		transforms.CenterCrop(size),
		transforms.ToTensor(),
		transforms.Normalize(means, stds)
	])

	dataset = dataset_imagenetvalpart(
		data_dir,
		xmllabeldir,
		synsetfile,
		maxnum,
		transform=transform
	)

	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=True
	)

	return dataloader


if __name__ == '__main__':
	from guidedbpcodehelpers import imshow2

	dataloader = get_Dataloader()

	inputs = next(iter(dataloader))

	image = inputs["image"]
	file_name = inputs["filename"]

	print(file_name)
	# print(image.shape)
	imshow2(image, image)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import random
from collections import namedtuple
import numpy as np

import dataset as dataset
import util as util
import net as net
from tqdm import tqdm

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

cate_list = ['StripedSkunk', 'Deer', 'Bobcat', 'Armadillo', 'FlyingSquirrel', 'Raccoon', 'Squirrel',
			 'Coyote', 'SpottedSkunk', 'Chipmunk', 'Moutainlion', 'Dog', 'FeralHog', 'Opossum',
			 'Mouse', 'Bird', 'Rabbit', 'Blackbear', 'Fox']

WORK_DIR  = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_predictions_test(model, iterator, device):

	model.eval()

	probs = []

	with torch.no_grad():
		for (x) in iterator:
			x = x.to(device)
			y_pred, _ = model(x)
			y_prob = F.softmax(y_pred, dim = -1)
			probs.append(y_prob.cpu())

	probs = torch.cat(probs, dim = 0)

	return probs

def get_classifier():
	ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
	resnet50_config = ResNetConfig(block = net.Bottleneck,
								n_blocks = [3, 4, 6, 3],
								channels = [64, 128, 256, 512])
	OUTPUT_DIM = len(cate_list)

	model = net.ResNet(resnet50_config, OUTPUT_DIM)
	model_path = os.path.join(WORK_DIR, 'MODEL', 'RESNET_50.pt')

	if torch.cuda.is_available() == False:
		model.load_state_dict(torch.load(model_path,map_location='cpu'))
	else:
		model.load_state_dict(torch.load(model_path))

	model = model.to(device)

	return model

def do_testing(CROP_DIR, model):
	data_dir_te  = CROP_DIR
	data_list_te = [f for f in os.listdir(data_dir_te) if 
					os.path.isfile(os.path.join(data_dir_te,f)) and f.endswith('.JPG')]

	class_dict = {}
	if len(data_list_te) == 0:
		return class_dict

	pretrained_size  = 224
	pretrained_means = [0.485, 0.456, 0.406]
	pretrained_stds  = [0.229, 0.224, 0.225]

	test_transforms = transforms.Compose([
							transforms.Resize(pretrained_size),
							transforms.CenterCrop(pretrained_size),
							transforms.ToTensor(),
							transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
						])

	test_data = dataset.Spotted_Skunk_Test(data_dir = data_dir_te,
										  data_list = data_list_te,
										  cate_list = cate_list,
										  transform = test_transforms)

	BATCH_SIZE = 4
	test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)

	probs = get_predictions_test(model, test_iterator, device)
	pred_labels = torch.argmax(probs, 1)

	for i in range(len(data_list_te)):
		class_dict[data_list_te[i]] = str(pred_labels[i].item()) + ' ' + str(max(probs[i][:]).item())

	return class_dict


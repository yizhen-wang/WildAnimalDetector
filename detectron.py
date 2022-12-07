import os
import cv2
import torch
import torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from imantics import Mask
from PIL import Image
from skimage import io

WORK_DIR  = os.getcwd()
CATE_LIST = ['StripedSkunk', 'Deer', 'Bobcat', 'Armadillo', 'FlyingSquirrel', 'Raccoon', 'Squirrel',
			 'Coyote', 'SpottedSkunk', 'Chipmunk', 'Moutainlion', 'Dog', 'FeralHog', 'Opossum', 'Mouse',
			 'Bird', 'Rabbit', 'Blackbear', 'Fox']

regist_dir = os.path.join(WORK_DIR, 'REGIST')
register_coco_instances('Spotted_Skunk_Data1', {}, os.path.join(regist_dir, 'regist_D1.json'), regist_dir)
metadata = MetadataCatalog.get('Spotted_Skunk_Data1')
metadata.thing_classes = CATE_LIST

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(WORK_DIR, 'MODEL', 'mask_rcnn_R_50_FPN_3x.pth')
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor = DefaultPredictor(cfg)

def do_segmentation(img_path, img_crop_dir):
	img_name = img_path.split('/')[-1].split('.')[0]

	img = io.imread(img_path)
	image_h = img.shape[0]
	image_w = img.shape[1]

	im = cv2.imread(img_path)
	outputs = predictor(im)
	score_list   = outputs["instances"].scores.tolist()
	pred_classes = outputs["instances"].pred_classes.tolist()
	pred_masks   = outputs["instances"].pred_masks

	bboxes_dict = {}

	for i in range(len(pred_classes)):
		mask = torch.Tensor.cpu(pred_masks).detach().numpy()[i][:][:].astype(int)
		bbox_info = list(Mask(mask).bbox().bbox())

		xmin = int(bbox_info[0])
		xmax = int(bbox_info[2])
		ymin = int(bbox_info[1])
		ymax = int(bbox_info[3])

		if img_crop_dir != None:
			cropped_img_name = img_name+'_'+str(i)+'.JPG'
			cropped_img_path = os.path.join(img_crop_dir, cropped_img_name)
			crop_img  = img[ymin:ymax, xmin:xmax, :]
			crop_img_ = Image.fromarray(crop_img)
			crop_img_.save(cropped_img_path)
		else:
			cropped_img_name = img_name+'_'+str(i)

		bboxes_dict[cropped_img_name] = str(xmin/image_w)+' '+str(ymin/image_h)+' '+str(xmax/image_w)+' '+ \
										str(ymax/image_h)+' '+str(pred_classes[i])+' '+str(score_list[i])

	return bboxes_dict




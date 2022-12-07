import os
import cv2
import json
import argparse
import numpy as np
from ensemble_boxes import *
from skimage import io

import detectron as det
import classification as cla

import warnings
warnings.filterwarnings("ignore")


CATE_LIST = ['StripedSkunk', 'Deer', 'Bobcat', 'Armadillo', 'FlyingSquirrel', 'Raccoon', 'Squirrel',
			 'Coyote', 'SpottedSkunk', 'Chipmunk', 'Moutainlion', 'Dog', 'FeralHog', 'Opossum', 'Mouse',
			 'Bird', 'Rabbit', 'Blackbear', 'Fox']

bboxes_info_dict = {}

def do_detection(img_path, args):
	print(img_path)
	img_name = img_path.split('/')[-1].split('.')[0]

	print('--------------------------')
	print('Do Segmentation')
	print('--------------------------')
	if args.Classification_Usage == 'True':
		img_crop_dir = os.path.join(args.Output_Dir, 'CROP_IMG', img_name)
		os.makedirs(img_crop_dir, exist_ok=True)
	else:
		img_crop_dir = None

	bboxes_info_dict[img_name] = det.do_segmentation(img_path, img_crop_dir)
	print(bboxes_info_dict[img_name])

	if args.Classification_Usage == 'True':
		print('--------------------------')
		print('Do Classification')
		print('--------------------------')
		classifier = cla.get_classifier()
		class_dict = cla.do_testing(img_crop_dir, classifier)

		for key in class_dict.keys():
			bboxes_info_dict[img_name][key] = bboxes_info_dict[img_name][key] + ' ' + str(class_dict[key])
		print(bboxes_info_dict[img_name])


	if args.WBF_Usage == 'True':
		print('--------------------------')
		print('Do ENSEMBLE WBF')
		print('--------------------------')
		boxes_list  = []
		scores_list = []
		labels_list = []

		for key in bboxes_info_dict[img_name].keys():
			splits = list(map(float, bboxes_info_dict[img_name][key].split()))
			
			boxes_list.append([splits[0], splits[1], splits[2], splits[3]])
			scores_list.append(splits[5])

			if args.Classification_Usage == 'True':
				det_conf = splits[5]
				cla_conf = splits[7]
				if det_conf < cla_conf:
					labels_list.append(int(splits[6]))
				else:
					labels_list.append(int(splits[4]))
			else:
				labels_list.append(int(splits[4]))
	
		boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, iou_thr=0.8, skip_box_thr=0.0)
		bboxes_info_dict_new = {}
		for i in range(len(boxes)):
			bboxes_info_dict_new[i] = str(round(boxes[i][0],4)) + ' ' + str(round(boxes[i][1],4)) + ' ' + str(round(boxes[i][2],4)) + ' ' + \
										str(round(boxes[i][3],4)) + ' ' + str(int(labels[i])) + ' ' + str(round(scores[i],4))
		print(bboxes_info_dict_new)
		bboxes_info_dict[img_name] = bboxes_info_dict_new

	if args.Draw_Detection == 'True':
		out_img = os.path.join(args.Output_Dir, 'IMG_WITH_BBOX', img_path.split('/')[-1])

		if not os.path.exists(out_img):
			img = io.imread(img_path)
			image_h = img.shape[0]
			image_w = img.shape[1]

			img = cv2.imread(img_path)
			for key in bboxes_info_dict[img_name].keys():
				splits = list(map(float, bboxes_info_dict[img_name][key].split()))
				xmin = int(float(splits[0]) * image_w)
				ymin = int(float(splits[1]) * image_h)
				xmax = int(float(splits[2]) * image_w)
				ymax = int(float(splits[3]) * image_h)
				cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 5)
			cv2.imwrite(out_img, img)


def gen_result(args):
	output_dict = {}
	for img_name in bboxes_info_dict:
		output_dict[img_name] = []
		detect_idx = 1
		for key in bboxes_info_dict[img_name].keys():
			splits = list(map(float, bboxes_info_dict[img_name][key].split()))
			detect_dict = {}
			detect_dict['id'] = detect_idx
			detect_dict['bbox'] = [[splits[0], splits[1], splits[2], splits[3]]]


			if args.WBF_Usage == 'True':
				detect_dict['class'] = CATE_LIST[int(splits[4])]
				detect_dict['conf'] = splits[5]
			elif args.Classification_Usage == 'True':
				det_conf = splits[5]
				cla_conf = splits[7]
				if det_conf < cla_conf:
					detect_dict['class'] = CATE_LIST[int(splits[6])]
					detect_dict['conf'] = splits[7]
				else:
					detect_dict['class'] = CATE_LIST[int(splits[4])]
					detect_dict['conf'] = splits[5]
			else:
				detect_dict['class'] = CATE_LIST[int(splits[4])]
				detect_dict['conf'] = splits[5]
		output_dict[img_name].append(detect_dict)

	result_file = os.path.join(args.Output_Dir, 'detection.json')
	with open(result_file, 'w') as f:
		json.dump(output_dict, f, indent=4)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Animal Detection')
	parser.add_argument('--ImgPath', type=str, help='The path of image file.')
	parser.add_argument('--DirPath', type=str, help='The path of image dir.')
	parser.add_argument('--Classification_Usage', type=str, default='False', help='If classification model will be used to help detection.')
	parser.add_argument('--WBF_Usage', type=str, default='False', help='If Weighted-Boxes-Fusion will be used to help detection.')
	parser.add_argument('--Output_Dir', type=str, default= os.path.join(os.getcwd(), 'Result'), 
							help='The path of output.')
	parser.add_argument('--Draw_Detection', type=str, default= 'False',
							help='If generate images with detected animal.')

	args = parser.parse_args()
	#print(args)
	os.makedirs(args.Output_Dir, exist_ok=True)

	if args.Draw_Detection == 'True':
		img_dir_with_bbox = os.path.join(args.Output_Dir, 'IMG_WITH_BBOX')
		os.makedirs(img_dir_with_bbox, exist_ok=True)

	if args.ImgPath != None:
		img_path = args.ImgPath
		do_detection(img_path, args)
	elif args.DirPath != None:
		img_dir = args.DirPath
		image_list = [f for f in os.listdir(img_dir) if 
						os.path.isfile(os.path.join(img_dir, f)) and f.endswith('.jpg') or f.endswith('.JPG')]
		if len(image_list) == 0:
			print('Your image folder is empty, please make sure your image file is end with .jpg or .JPG')
			exit()
		else:
			for image in image_list:
				img_path = os.path.join(img_dir, image)
				do_detection(img_path, args)
	else:
		print('Needs path of image or path of image dir, use -h for more imformation')
		exit()

	gen_result(args)
import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('xyz_file', help='your XYZ file')
parser.add_argument('-e', '--edge', action='store_true', help='activate edge detection (default: false)')
parser.add_argument('-k', '--kernel', type=int, default=3, help='kernel size of morphological transformation (default: 3)')
parser.add_argument('-p', '--prefix', default='', help='prefix of output files')
parser.add_argument('-r', '--resolution', type=int, default=500, help='resolution of output images (default: 500)')
parser.add_argument('-s', '--slices', type=int, default=100, help='number of slices (default: 100)')
args = parser.parse_args()

input_filename = args.xyz_file
edge_detection = args.edge
kernel_size = args.kernel
n_slices = args.slices
output_resolution = args.resolution
output_prefix = args.prefix

xyz_file = open(input_filename, 'r')
list = xyz_file.read().split()
xyz_file.close()

x_list = []
y_list = []
z_list = []

for x in range(0, len(list), 3):
	x_list.append(float(list[x]))

for y in range(1, len(list), 3):
	y_list.append(float(list[y]))

for z in range(2, len(list), 3):
	z_list.append(float(list[z]))

len_list = len(z_list)

x_max = max(x_list)
x_min = min(x_list)
x_mid = (x_max + x_min) / 2
x_len = x_max - x_min

y_max = max(y_list)
y_min = min(y_list)
y_mid = (y_max + y_min) / 2
y_len = y_max - y_min

z_max = max(z_list)
z_min = min(z_list)

d = (z_max - z_min) / n_slices
w = output_resolution
h = int((y_max - y_min) / (x_max - x_min) * w)

px = x_len / w # [m/px]
px2 = px * px  # [m2/px2]

kernel = np.ones((kernel_size, kernel_size), np.uint8)

print('Processing {}\n  Width = {}[m]\n  Height = {}[m]\n  Pixel length = {}[m/px]\n  Pixel area = {}[m2/px2]\n  Thikness of slice = {}[m]'.format(input_filename, x_len, y_len, px, px2, d))

for level in tqdm(range(0, n_slices)):
	blank_image = np.zeros((h, w, 3))
	# blank_image += 255
	for i in range(0, len_list):
		base = z_min + d * level
		z = z_list[i]
		if (z >= base and z < base + d):
			x = int((x_list[i] - x_mid) / x_len * w + w / 2)
			y = int((y_list[i] - y_mid) / y_len * h + h / 2)
			cv2.circle(img=blank_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
	result_image = cv2.morphologyEx(blank_image, cv2.MORPH_CLOSE, kernel)
	if edge_detection:
		gray_image = cv2.cvtColor(np.uint8(result_image), cv2.COLOR_BGR2GRAY)
		contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(result_image, contours, -1, (0, 0, 255), 3)
	filename = '{}{:04d}.png'.format(output_prefix, level)
	cv2.imwrite(filename, result_image)

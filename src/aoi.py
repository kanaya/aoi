import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageFilter

parser = argparse.ArgumentParser()
parser.add_argument('xyz_file', help='your XYZ file')
parser.add_argument('-p', '--prefix', default='', help='prefix of output files')
parser.add_argument('-r', '--resolution', type=int, default=500, help='resolution of output images (default: 500)')
parser.add_argument('-s', '--slices', type=int, default=100, help='number of slices (default: 100)')
args = parser.parse_args()

xyzfile = open(args.xyz_file, 'r')
list = xyzfile.read().split()
xyzfile.close()

output_resolution = args.resolution
output_prefix = args.prefix
n_slices = args.slices

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

y_max = max(y_list)
y_min = min(y_list)
y_mid = (y_max + y_min) / 2

z_max = max(z_list)
z_min = min(z_list)

d = (z_max - z_min) / n_slices
w = output_resolution
h = int((y_max - y_min) / (x_max - x_min) * w)

print('Processing {} slices'.format(n_slices), end='', flush=True, file=sys.stderr)

for level in range(0, n_slices):
	if (level % 10 == 0):
		print('.', end='', flush=True, file=sys.stderr)
	blank = np.zeros((h, w, 3))
	blank += 255
	for i in range(0, len_list):
		base = z_min + d * level
		z = z_list[i]
		if (z >= base and z < base + d):
			x = int((x_list[i] - x_mid) / (x_max - x_min) * w + w / 2)
			y = int((y_list[i] - y_mid) / (y_max - y_min) * h + h / 2)
			cv2.circle(img=blank, center=(x, y), radius=1, color=(0,0,0), thickness=-1)
	filename = '{}F_{:04d}.png'.format(output_prefix, level)
	cv2.imwrite(filename, blank)

print('done', file=sys.stderr)

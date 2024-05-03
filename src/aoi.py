import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('xyz_file', help='your XYZ file')
parser.add_argument('-a', '--area', action='store_true', help='calculate area (default: false)')
parser.add_argument('-e', '--edge', action='store_true', help='activate edge detection (default: false)')
parser.add_argument('-k', '--kernel', type=int, default=3, help='kernel size of morphological transformation (default: 3)')
parser.add_argument('-p', '--prefix', default='', help='prefix of output files')
parser.add_argument('-r', '--resolution', type=int, default=500, help='resolution of output images (default: 500)')
parser.add_argument('-s', '--slices', type=int, default=100, help='number of slices (default: 100)')
parser.add_argument('-z', '--zstart', type=int, default=0, help='starting plane (default: 0)')
args = parser.parse_args()

input_filename = args.xyz_file
calc_area = args.area
edge_detection = args.edge or args.area
kernel_size = args.kernel
n_slices = args.slices
output_resolution = args.resolution
output_prefix = args.prefix
z_start = args.zstart

x_list = []
y_list = []
z_list = []

area = [0] * n_slices

def main():
	print('# Given parameters are:\n'
		'#   --area {}\n'
		'#   --edge {}\n'
		'#   --kernel {}\n'
		'#   --resolution {}\n'
		'#   --slices {}\n'
		'#   --zstart {}'
		.format(calc_area, edge_detection, kernel_size, output_resolution, n_slices, z_start))

	print('Reading {}... '.format(input_filename), end='', file=sys.stderr)
	xyz_file = open(input_filename, 'r')
	xyz_list = xyz_file.read().split()
	xyz_file.close()
	print('done', file=sys.stderr)

	for x in range(0, len(xyz_list), 3):
		x_list.append(float(xyz_list[x]))
	for y in range(1, len(xyz_list), 3):
		y_list.append(float(xyz_list[y]))
	for z in range(2, len(xyz_list), 3):
		z_list.append(float(xyz_list[z]))

	x_max = max(x_list)
	x_min = min(x_list)
	x_mid = (x_max + x_min) / 2
	x_dif = x_max - x_min

	y_max = max(y_list)
	y_min = min(y_list)
	y_mid = (y_max + y_min) / 2
	y_dif = y_max - y_min

	z_max = max(z_list)
	z_min = min(z_list)
	z_dif = z_max - z_min

	d = z_dif / n_slices
	w = output_resolution
	h = int(y_dif / x_dif * w)

	px = x_dif / w # [m/px]
	px2 = px * px  # [m2/px2]

	print('# Processing {}\n'
		'#   Size of Easting = {:,.3f}[m]\n'
		'#   Size of Northing = {:,.3f}[m]\n'
		'#   Z_min = {:,.3f}[m]\n'
		'#   Z_max = {:,.3f}[m]\n'
		'#   Pixel length = {:,.3f}[m/px]\n'
		'#   Pixel area = {:,.3f}[m²/px²]\n'
		'#   Thikness of slice = {:,.3f}[m]'
		.format(input_filename, x_dif, y_dif, z_min, z_max, px, px2, d))
	print('# If you want to make a voxel being 100[mm] cube, try the following parameter.\n'
		'#   --resolution {} --slices {}'
		.format(int(x_dif / 0.1), int(z_dif / 0.1)))

	# for slice in tqdm(range(z_start, n_slices)):
	#	process_slice(slice, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d)
	m = map(lambda s: process_slice(s, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d), range(z_start, n_slices))
	results = list(m)
	if calc_area:
		total_area = sum(area)
		print('total area is {:,.3f}, meaning {:,.3f}[m³]'.format(total_area, total_area * px2 * d))

def process_slice(slice, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	blank_image = np.zeros((h, w, 3))
	len_list = len(z_list)
	for i in range(0, len_list):
		base = z_min + d * slice
		z = z_list[i]
		if (z >= base and z < base + d):
			x = int((x_list[i] - x_mid) / x_dif * w + w / 2)
			y = int((y_list[i] - y_mid) / y_dif * h + h / 2)
			cv2.circle(img=blank_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
	result_image = cv2.morphologyEx(blank_image, cv2.MORPH_CLOSE, kernel)
	if edge_detection:
		gray_image = cv2.cvtColor(np.uint8(result_image), cv2.COLOR_BGR2GRAY)
		contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(result_image, contours, -1, (0, 0, 255), 3)
		cv2.fillPoly(result_image, contours, (0, 0, 255))
		if calc_area:
			for contour in contours:
				area[slice] = cv2.contourArea(contour)
	filename = '{}{:04d}.png'.format(output_prefix, slice)
	cv2.imwrite(filename, result_image)

if __name__ == '__main__':
	main()

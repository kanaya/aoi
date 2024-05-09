import argparse
import concurrent.futures
import cv2
import functools
import multiprocessing as mp
import numpy as np
import sys
from PIL import Image, ImageFilter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('xyz_file', help='your XYZ file')
parser.add_argument('-a', '--area', action='store_true', help='calculate area (default: false)')
parser.add_argument('-c', '--cutoff', type=float, default=0.0, help='cut off of z-axis (default: 0.0)')
parser.add_argument('-e', '--edge', action='store_true', help='activate edge detection (default: false)')
parser.add_argument('-k', '--kernel', type=int, default=3, help='kernel size of morphological transformation (default: 3)')
parser.add_argument('-m', '--multiprocess', action='store_true', help='enable multiprocessing (default: false) [EXPERIMENTAL]')
parser.add_argument('-p', '--prefix', default='', help='prefix of output files')
parser.add_argument('-r', '--resolution', type=int, default=500, help='resolution of output images (default: 500)')
parser.add_argument('-s', '--slices', type=int, default=100, help='number of slices (default: 100)')
parser.add_argument('-w', '--warning', action='store_true', help='show warning if there is strange calculation')
parser.add_argument('-z', '--zstart', type=int, default=0, help='starting plane (default: 0)')
args = parser.parse_args()

input_filename = args.xyz_file
calc_area = args.area or args.warning
edge_detection = args.edge or args.area or args.warning
kernel_size = args.kernel
multiprocess = args.multiprocess
output_prefix = args.prefix
output_resolution = args.resolution
n_slices = args.slices
warning_option = args.warning
z_cutoff = args.cutoff
z_start = args.zstart

x_list = []
y_list = []
z_list = []

area = [0] * n_slices

def main():
	print('# Given parameters are:\n'
		'#   --area {}\n'
		'#   --cutoff {}\n'
		'#   --edge {}\n'
		'#   --kernel {}\n'
		'#   --resolution {}\n'
		'#   --slices {}\n'
		'#   --warning {}\n'
		'#   --zstart {}'
		.format(calc_area, z_cutoff, edge_detection, kernel_size, output_resolution, n_slices, warning_option, z_start))

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
		'#   Z_cutoff = {:,.3f}[m]\n'
		'#   Z_max = {:,.3f}[m]\n'
		'#   Pixel length = {:,.3f}[m/px]\n'
		'#   Pixel area = {:,.3f}[m²/px²]\n'
		'#   Thikness of slice = {:,.3f}[m]'
		.format(input_filename, x_dif, y_dif, z_min, z_cutoff, z_max, px, px2, d))
	print('# If you want to make a voxel being 100[mm] cube, try the following parameter.\n'
		'#   --resolution {} --slices {}'
		.format(int(x_dif / 0.1), int(z_dif / 0.1)))

	if multiprocess:
		shared_x_list = ndarray_to_shared_memory(np.array(x_list, dtype=np.float64))
		shared_y_list = ndarray_to_shared_memory(np.array(y_list, dtype=np.float64))
		shared_z_list = ndarray_to_shared_memory(np.array(z_list, dtype=np.float64))
		with concurrent.futures.ThreadPoolExecutor(initializer=lambda: globals().update(dict(shared_x_list=shared_x_list, shared_y_list=shared_y_list, shared_z_list=shared_z_list))) as executor:
			iter = range(z_start, n_slices)
			m = executor.map(functools.partial(process_slice, x_list, y_list, z_list, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d), iter)
			# m = executor.map(process_slice_mp_params, [(x_mid, y_mid, x_dif, y_dif, z_min, w, h, d, slice) for slice in iter])
			results = list(m)
			# *** multithread version ***
			# with concurrent.futures.ThreadPoolExecutor() as executor:
			#   iter = range(z_start, n_slices)
			#   m = executor.map(lambda s: process_slice(x_list, y_list, z_list, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d, s), iter)
			#   results = list(m)
	else:
		m = map(lambda s: process_slice(x_list, y_list, z_list, x_mid, y_mid, x_dif, y_dif, z_min, z_cutoff, w, h, d, s), tqdm(range(z_start, n_slices)))
		results = list(m)
		# *** for version ***
		# for slice in tqdm(range(z_start, n_slices)):
		# 	process_slice(x_list, y_list, z_list, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d, slice)

	if calc_area:
		total_area = sum(area)
		print('total area is {:,.3f}, meaning {:,.3f}[m³]'.format(total_area, total_area * px2 * d))
	if warning_option:
		show_warning_if_exists_strange_calculation(area)

# def process_slice_mp_params(a):
# 	process_slice_mp(*a)

def process_slice_mp(x_mid, y_mid, x_dif, y_dif, z_min, w, h, d, slice):
	x_list = shared_memory_to_ndarray(shared_x_list)
	y_list = shared_memory_to_ndarray(shared_y_list)
	z_list = shared_memory_to_ndarray(shared_z_list)
	process_slice(x_list, y_list, z_list, x_mid, y_mid, x_dif, y_dif, z_min, w, h, d, slice)

def process_slice(x_list, y_list, z_list, x_mid, y_mid, x_dif, y_dif, z_min, z_cutoff, w, h, d, slice):
	global edge_detection
	global calc_area
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	blank_image = np.zeros((h, w, 3))
	len_list = len(z_list)
	for i in range(0, len_list):
		base = z_min + d * slice
		if (base >= z_cutoff):
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

# The following codes are copied from https://zenn.dev/kzm4269/articles/80df87e6e9001f
def ndarray_to_shared_memory(data):
	data = np.asarray(data)
	value_type = np.ctypeslib.as_ctypes_type(np.uint8)
	if data.ndim >= 1:
		value_type *= data.itemsize * data.shape[-1]
	for dim in data.shape[:-1][::-1]:
		value_type *= dim
	value = mp.RawValue(value_type)
	try:
		np.ctypeslib.as_array(value)[:] = data.view(np.uint8)
	except TypeError:
		raise TypeError(f'unsupported dtype: {data.dtype!r}')
	return value, data.dtype

def shared_memory_to_ndarray(data):
	value, dtype = data
	return np.ctypeslib.as_array(value).view(dtype)

def show_warning_if_exists_strange_calculation(area):
	for i in range(1, len(area) - 1):
		a0 = area[i - 1]
		a1 = area[i]
		a2 = area[i + 1]
		aa = (a0 + a2) / 2.0
		ar = a1 / aa
		if ar < 0.8 or ar > 1.2:
			print('Area of slice {} looks strange.'.format(i), file=sys.stderr)

if __name__ == '__main__':
	main()

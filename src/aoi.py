import numpy as np
from PIL import Image, ImageFilter
import cv2

myfile = open('menkaure-pyramid-1000th.xyz', 'r')
list = myfile.read().split()
myfile.close()

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

print('x_max={}, x_min={}, dx={}'.format(x_max, x_min, x_max-x_min))
print('y_max={}, y_min={}, dy={}'.format(y_max, y_min, y_max-y_min))
print('z_max={}, z_min={}, dz={}'.format(z_max, z_min, z_max-z_min))

n_slices = 100
d = (z_max - z_min) / n_slices
w = 500
h = int((y_max - y_min) / (x_max - x_min) * w)

print("h={}".format(h))

for level in range(0, n_slices):
	blank = np.zeros((h, w, 3))
	blank += 255
	for i in range(0, len_list):
		base = z_min + d * level
		z = z_list[i]
		if (z >= base and z < base + d):
			x = int((x_list[i] - x_mid) / (x_max - x_min) * w + w / 2)
			y = int((y_list[i] - y_mid) / (y_max - y_min) * h + h / 2)
			cv2.circle(img=blank, center=(x, y), radius=1, color=(0,0,0), thickness=-1)
	filename = 'result/F_{}-{}.png'.format(level, level + 1)
	cv2.imwrite(filename, blank)

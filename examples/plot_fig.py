import numpy as np
import matplotlib.pyplot as plt
import shapely as shp
from scipy import spatial

line = np.load('./examples/rl_race/f1tenth_racetracks/Barcelona/centerline.npy', allow_pickle=True)

line = line[:, 0:2]
track_poly = shp.Polygon(line)
track_xy_left = track_poly.buffer(0.93)
track_xy_right = track_poly.buffer(-0.93)
track_xy_left = np.array(track_xy_left.exterior.coords)
track_xy_right = np.array(track_xy_right.exterior.coords)
left_tree = spatial.KDTree(track_xy_left)
right_tree = spatial.KDTree(track_xy_right)
line_i = []
line_o = []
for i in range(line.shape[0]):
    min_d, min_index = left_tree.query(line[i])
    line_i.append(track_xy_left[min_index])
    min_d, min_index = right_tree.query(line[i])
    line_o.append(track_xy_right[min_index])
line_i = np.array(line_i)
line_o = np.array(line_o)

with open('track.json', 'a') as f:
    f.write('{'+'\n')
    f.write('\"X\": [')
for i in range(line.shape[0]):
    with open('track.json', 'a') as f:
        if i < line.shape[0]-1:
            f.write(str(line[i][0])+', ')
        else:
            f.write(str(line[i][0])+'],\n')
with open('track.json', 'a') as f:
    f.write('\"Y\": [')
for i in range(line.shape[0]):
    with open('track.json', 'a') as f:
        if i < line.shape[0]-1:
            f.write(str(line[i][1])+', ')
        else:
            f.write(str(line[i][1])+'],\n')

with open('track.json', 'a') as f:
    f.write('\"X_i\": [')
for i in range(line_i.shape[0]):
    with open('track.json', 'a') as f:
        if i < line_i.shape[0]-1:
            f.write(str(line_i[i][0])+', ')
        else:
            f.write(str(line_i[i][0])+'],\n')
with open('track.json', 'a') as f:
    f.write('\"Y_i\": [')
for i in range(line_i.shape[0]):
    with open('track.json', 'a') as f:
        if i < line_i.shape[0]-1:
            f.write(str(line_i[i][1])+', ')
        else:
            f.write(str(line_i[i][1])+'],\n')

with open('track.json', 'a') as f:
    f.write('\"X_o\": [')
for i in range(line_o.shape[0]):
    with open('track.json', 'a') as f:
        if i < line_o.shape[0]-1:
            f.write(str(line_o[i][0])+', ')
        else:
            f.write(str(line_o[i][0])+'],\n')
with open('track.json', 'a') as f:
    f.write('\"Y_o\": [')
for i in range(line_o.shape[0]):
    with open('track.json', 'a') as f:
        if i < line_o.shape[0]-1:
            f.write(str(line_o[i][1])+', ')
        else:
            f.write(str(line_o[i][1])+']\n')
with open('track.json', 'a') as f:
    f.write('}\n')

figure = plt.figure()
axes = figure.add_subplot(1,1,1)
axes.plot(line_i[:,0], line_i[:,1], 'k-')
# axes.plot(line_o[:,0], line_o[:,1], 'k-')
axes.plot(line[:, 0], line[:, 1], 'r--')
print(line.shape, track_xy_left.shape, track_xy_right.shape)

axes.legend()
plt.xlabel('X/m')
plt.ylabel('Y/m')
axes.set_aspect(1.0)
plt.show()
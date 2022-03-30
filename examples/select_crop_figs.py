import os
import datetime
import shutil
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch

import sunpy.map
from sunpy.map.sources.source_type import source_stretch
from sunpy.physics.differential_rotation import solar_rotate_coordinate

# center_coord = None

def onclick(event):
    # get clicked pixel coordinates
    x = event.xdata
    y = event.ydata
    print('click: xdata=%f, ydata=%f' % (x, y))
    
    global center_coord
    # transform the pixel to world (solar surface) coordinates
    center_coord = map_one.pixel_to_world(x*u.pix, y*u.pix)

in_directory = "../data/continuous_5min"
out_directory = "../data/continuous_5min_figs"
if not os.path.exists(out_directory):
    try:
        os.mkdir(out_directory)
    except:
        pass
maps = sorted(glob.glob(in_directory + "/*.fits"))


map_one = sunpy.map.Map(maps[0])
# image normalization used in aiamap.peek()
# https://docs.sunpy.org/en/stable/_modules/sunpy/map/sources/sdo.html
# it might be changed when you use another instrument
map_one.plot_settings['norm'] = ImageNormalize(
    stretch=source_stretch(map_one.meta, AsinhStretch(0.01)), clip=False)
fig_ = plt.figure(figsize=(10,10))
ax_ = fig_.add_subplot(1,1,1, projection = map_one)
cid = fig_.canvas.mpl_connect('button_press_event', onclick)
map_one.plot()
plt.show()


c_lon = int(center_coord.Tx.value)
c_lat = int(center_coord.Ty.value)

# print(f"last clicked coordinates are x:{c_lon} y:{c_lat}")
# print("please input the height of movie (unit: arcsec)")
# height = int(input())
# print("please input the width of movie (unit: arcsec)")
# width = int(input())
height, width = 500, 500 # DEBUG

fig_ = plt.figure(figsize=(10,10))
blc = SkyCoord(Tx=(c_lon-int(width/2))*u.arcsec,Ty=(c_lat-int(height/2))*u.arcsec,frame=map_one.coordinate_frame)
trc = SkyCoord(Tx=(c_lon+int(width/2))*u.arcsec,Ty=(c_lat+int(height/2))*u.arcsec,frame=map_one.coordinate_frame)
# print(blc, trc)
cropped_map = map_one.submap(blc, top_right=trc)
ax_ = fig_.add_subplot(1,1,1, projection = cropped_map)
cropped_map.plot()
plt.show()



# fig = plt.figure(figsize=(10, 10))
# map_init = sunpy.map.Map(maps[0])
# map_init.plot_settings['norm'] = ImageNormalize(
#     stretch=source_stretch(map_one.meta, AsinhStretch(0.01)), clip=False)
# blc = SkyCoord(Tx=(c_lon-int(width/2))*u.arcsec,Ty=(c_lat-int(height/2))*u.arcsec,frame=map_one.coordinate_frame)
# trc = SkyCoord(Tx=(c_lon+int(width/2))*u.arcsec,Ty=(c_lat+int(height/2))*u.arcsec,frame=map_one.coordinate_frame)
# map_init_crop = map_init.submap(blc, top_right=trc)
# ax = fig.add_subplot(1,1,1, projection = map_init_crop)
# map_init_crop.plot(axes = ax)
# plt.savefig(out_directory + "/")

for m in maps:
    fig = fig = plt.figure(figsize=(10, 10))
    map_iter = sunpy.map.Map(m)
    map_iter.plot_settings['norm'] = ImageNormalize(
        stretch=source_stretch(map_one.meta, AsinhStretch(0.01)), clip=False)
    mdata = map_iter.data
    ex = float(map_iter.meta['exptime'])
    # print(map_iter.meta['exptime'])
    if np.any(np.isnan(mdata)):
        map_iter.data = np.where(np.isnan(mdata), 0, mdata)/ex
    coord_iter = solar_rotate_coordinate(center_coord, time=map_iter.date)
    c_lon_n = int(coord_iter.Tx.value)
    c_lat_n = int(coord_iter.Ty.value)
    blc = SkyCoord(Tx=(c_lon_n-int(width/2))*u.arcsec,Ty=(c_lat_n-int(height/2))*u.arcsec,frame=map_iter.coordinate_frame)
    trc = SkyCoord(Tx=(c_lon_n+int(width/2))*u.arcsec,Ty=(c_lat_n+int(height/2))*u.arcsec,frame=map_iter.coordinate_frame)
    m_crop = map_iter.submap(blc, top_right=trc)
    ax = fig.add_subplot(1,1,1, projection = m_crop)
    m_crop.plot(axes = ax)
    plt.title(f"AIA {m_crop.meta['wavelnth']} {m_crop.meta['date-obs'][:10]} {m_crop.meta['date-obs'][11:19]}")
    t = datetime.datetime.strptime(m_crop.meta['date-obs'][:19], '%Y-%m-%dT%H:%M:%S')
    t_str = datetime.datetime.strftime(t, '%y%m%d_%H%M%S')
    savename = f"/AIA_{m_crop.meta['wavelnth']}_{t_str}"
    plt.savefig(out_directory + savename)
    
# ani = animation.FuncAnimation(fig, update, interval=100, frames=len(maps), blit=True)
# ani.save(f"test_.mp4", writer='ffmpeg')
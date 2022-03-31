import datetime
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import astropy
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import sunpy
import sunpy.map
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.net import Fido, attrs as a
from sunpy.coordinates.sun import carrington_rotation_number
import pfsspy
import pfsspy.utils

# set your registered email to JSOC_EMAIL
from config import *

"""
should be fix:

- the observed times of downloaded hmi and aia tend to have gap,
so it would be better if one map is reprojected to be another map.
- about the value of header.cdelt2
- 
"""

decay_list = []

# button click and put marker function
def onclick(event):
    # get clicked pixel coordinates
    x = event.xdata
    y = event.ydata
    print('click: xdata=%f, ydata=%f' % (x, y))
    
    # transform the pixel to world (solar surface) coordinates
    coord = cropped_amap.pixel_to_world(x*u.pix, y*u.pix)
    
    # calibrate the difference of observed time
    # I should solve some warnings about the difference between "solar time" "Earth time"?
    coord_heligra = coord.transform_to(sunpy.coordinates.HeliographicCarrington)
    coord_syn = solar_rotate_coordinate(coord_heligra, time=smap.date)
    c_spix = smap.world_to_pixel(coord_syn)

    # with open(f"coord_{dt_str}.txt", mode="a+") as f:
    #     f.writelines(f"{c_spix[0].value},{c_spix[1].value}\n")

    ax4.plot_coord(coord, marker="+", linewidth=5, markersize=10)
    ax1.plot_coord(coord_syn, color="white", marker="+", linewidth=5, markersize=10)

    di_one, h_one = interp_decay(di, c_spix.x.value, c_spix.y.value, height, h_limit)
    decay_list.append(di_one)
    ax2.plot(h_one, di_one)

    di_ave = np.average(np.array(decay_list), axis=0)
    ax3.cla()
    ax3.set_ylim([0, 4.5])
    ax3.set_title("Averaged Decay Index")
    ax3.set_xlabel("height to solar surface [Mm]")
    ax3.set_ylabel("decay index")
    ax3.grid()
    ax3.plot(h_one, di_ave, color="black")
    plt.draw()


# calculate decay index
def cal_decay(pfss_out):
    b_theta = pfss_out.bg[:,:,:,1]
    b_phi = pfss_out.bg[:,:,:,0]
    # b_r = pfss_out.bg[:,:,:,2]
    bh = np.sqrt(b_theta*b_theta + b_phi*b_phi)
    h = (np.exp(pfss_out.grid.rg) - 1)[1:]

    dln_bh = np.diff(np.log(bh[:,:,1:]), axis=-1)
    dln_h = np.diff(np.log(h))
    di = -dln_bh/dln_h
    # transpose axis same as synoptic map
    di = di.transpose(1,0,2)
    # assume the index at solar surface to be same as the index one step above
    di = np.pad(di, [(0,0),(0,0),(1,0)], "edge")
    return di

# average 4 point of decay index and interpolate the height
# 'h_limit' is the maximal height to interpolate (unit: Mm)
def interp_decay(di, x, y, height, h_limit):
    h_limited = height[np.where(height<=h_limit)[0]]
    di_limited = di[:,:,np.where(height<=h_limit)[0]]
    di_averaged = np.average(di_limited[math.floor(y):math.floor(y)+2,math.floor(x):math.floor(x)+2], axis=(0,1))
    f = interpolate.interp1d(h_limited, di_averaged, kind="cubic")
    h_interp = np.linspace(0, math.floor(h_limited[-1]), 1000)
    di_interp = f(h_interp)
    return di_interp, h_interp



def get_data_local(num, wl, dateaia, datehmi):
    dta = datetime.datetime(*dateaia)
    dth = datetime.datetime(*datehmi)
    hmipath = f"/Users/kihara/Files/research/02_SEP_CDAW/SEP_D/data/{dic[num]}/hmi/hmi.m_45s.{dth.year:04}{dth.month:02}{dth.day:02}_{dth.hour:02}{dth.minute:02}*_TAI.2.magnetogram.fits"
    if wl in (1600, 1700):
        aiapath = f"/Users/kihara/Files/research/02_SEP_CDAW/SEP_D/data/{dic[num]}/aia{wl}/aia.lev1_uv_24s.{dta.year:04}-{dta.month:02}-{dta.day:02}T{dta.hour:02}{dta.minute:02}*Z.{wl}.image_lev1.fits"
    else:
        aiapath = f"/Users/kihara/Files/research/02_SEP_CDAW/SEP_D/data/{dic[num]}/aia{wl:03}/aia.lev1_euv_12s.{dta.year:04}-{dta.month:02}-{dta.day:02}T{dta.hour:02}{dta.minute:02}*.{wl}.image_lev1.fits"
    print(hmipath)
    print(aiapath)
    hp = glob.glob(hmipath)[0]
    ap = glob.glob(aiapath)[0]
    return hp, ap

def fido_get_aia_and_hmi(wl, dt):
    t_start = astropy.time.Time(datetime.datetime(*dt), scale='utc', format='datetime')
    q_aia = Fido.search(
        a.Time(t_start, t_start + 5 * u.min),
        a.Wavelength(wl * u.angstrom),
        a.Instrument.aia,
        # a.jsoc.Notify(JSOC_EMAIL),
    )
    d_aia = Fido.fetch(q_aia[0][0])

    q_hmi = Fido.search(
        a.Time(t_start, t_start + 5 * u.min),
        a.Instrument.hmi,
        a.Physobs.los_magetic_field,
        # a.jsoc.Notify(JSOC_EMAIL),
    )
    d_hmi = Fido.fetch(q_hmi[0][0])

    return d_aia, d_hmi

def fido_get_synoptic(dt):
    t_syn = astropy.time.Time(dt, scale='utc')
    car_num = carrington_rotation_number(t_syn)
    q_syn = Fido.search(
        a.Time(t_syn, t_syn),
        a.jsoc.Series('hmi.synoptic_mr_polfil_720s'),
        a.jsoc.PrimeKey('CAR_ROT', int(car_num)),
        a.jsoc.Notify(JSOC_EMAIL),
    )
    d_syn = Fido.fetch(q_syn[0][0])
    return d_syn

def onclick_small(event):
    # get clicked pixel coordinates
    x = event.xdata
    y = event.ydata
    print('click: xdata=%f, ydata=%f' % (x, y))
    
    global center_coord
    # transform the pixel to world (solar surface) coordinates
    center_coord = amap.pixel_to_world(x*u.pix, y*u.pix)

# am, hm = fido_get_aia_and_hmi(1600, (2012, 3, 14, 15, 21))
am, hm = fido_get_aia_and_hmi(1600, (2014, 10, 22, 14, 27))
hmap = sunpy.map.Map(hm).rotate(angle = 180.0*u.deg)
amap = sunpy.map.Map(am)

fig_pre = plt.figure(figsize=(9,9))
ax_pre = fig_pre.add_subplot(1,1,1, projection=amap)
cid = fig_pre.canvas.mpl_connect('button_press_event', onclick_small)
amap.plot(axes = ax_pre, clip_interval = (1, 99.99)*u.percent)
# levels = [-50, 50] * u.Gauss
# cset = hmap.draw_contours(levels, axes=ax_pre, cmap="bwr", alpha=0.7)
plt.show()
plt.close()

c_lon = int(center_coord.Tx.value)
c_lat = int(center_coord.Ty.value)

print(f"last clicked coordinates are x:{c_lon} y:{c_lat}")
print("please input the height of movie (unit: arcsec)")
height = int(input())
print("please input the width of movie (unit: arcsec)")
width = int(input())
# height, width = 500, 500 # DEBUG

fig_crop = plt.figure(figsize=(9,9))
blc_a = SkyCoord(Tx=(c_lon-int(width/2))*u.arcsec,Ty=(c_lat-int(height/2))*u.arcsec,frame=amap.coordinate_frame)
trc_a = SkyCoord(Tx=(c_lon+int(width/2))*u.arcsec,Ty=(c_lat+int(height/2))*u.arcsec,frame=amap.coordinate_frame)
blc_h = SkyCoord(Tx=(c_lon-int(width/2))*u.arcsec,Ty=(c_lat-int(height/2))*u.arcsec,frame=hmap.coordinate_frame)
trc_h = SkyCoord(Tx=(c_lon+int(width/2))*u.arcsec,Ty=(c_lat+int(height/2))*u.arcsec,frame=hmap.coordinate_frame)
# print(blc, trc)
cropped_amap = amap.submap(blc_a, top_right=trc_a)
cropped_hmap = hmap.submap(blc_h, top_right=trc_h)
ax_crop = fig_crop.add_subplot(1,1,1, projection = cropped_amap)
cropped_amap.plot()
levels = [-50, 50] * u.Gauss
cset = cropped_hmap.draw_contours(levels, axes=ax_crop, cmap="bwr", alpha=0.7)
plt.show()

# num = "024"
# hp, ap = get_data_local(num, 1700, (2011, 8, 4, 4, 1, 0), (2011, 8, 4, 3, 30, 0))

# num = "191"
# hp, ap = get_data_local(num, 1700, (2017, 7, 14, 1, 1, 0), (2017, 7, 14, 1, 1, 0))

# dt_str = str(datetime.datetime.now())
# with open(f"coord_{dt_str}.txt", mode="a+") as f:
#         f.writelines(f"{hmap}\n")
#         f.writelines(f"{amap}\n")

# data load
# car_num = carrington_rotation_number(amap.date)
# print(car_num)
# synoptic_url_small = f"http://jsoc.stanford.edu/data/hmi/synoptic/hmi.Synoptic_Mr_small.{int(car_num)}.fits"
# synoptic_url = f"http://jsoc.stanford.edu/data/hmi/synoptic/hmi.Synoptic_Mr.{int(car_num)}.fits"
# try:
#     synoptic_sample = download_file(synoptic_url_small, cache=False)
# except:
#     synoptic_sample = download_file(synoptic_url, cache=False)

smap = sunpy.map.Map(fido_get_synoptic(amap.date)).resample([720, 360] * u.pix)
# smap = open_syn_map(synoptic_sample).resample([720, 360] * u.pix)
# smap = open_syn_map(synoptic_2192)
print(smap.meta["CDELT2"])

# calc pfss
nr = 100
rss = 2.0 # source-surface height

pfss_in = pfsspy.Input(smap, nr, rss)
pfss_out = pfsspy.pfss(pfss_in)
ss_br = pfss_out.source_surface_br

# calculate decay index
di = cal_decay(pfss_out)
# translate the height to Mm from solar surface
height = (np.exp(pfss_out.grid.rg) - 1) * const.R_sun.to("Mm").value
h_limit = 200 # Mm



# prepare matplotlib figure and connect click function 
fig = plt.figure(figsize=(16, 8))
gs = matplotlib.gridspec.GridSpec(2,4)
cid = fig.canvas.mpl_connect('button_press_event', onclick)


# blc_hs = SkyCoord(cx*u.arcsec, cy*u.arcsec, frame=hmap.coordinate_frame)
# trc_hs = SkyCoord((cx+cs)*u.arcsec, (cy+cs)*u.arcsec, frame=hmap.coordinate_frame)
# blc_as = SkyCoord(cx*u.arcsec, cy*u.arcsec, frame=amap.coordinate_frame)
# trc_as = SkyCoord((cx+cs)*u.arcsec, (cy+cs)*u.arcsec, frame=amap.coordinate_frame)


# cutout AIA and HMI
# subh_s = hmap.submap(blc_hs, top_right = trc_hs)
# suba_s = amap.submap(blc_as, top_right = trc_as)

# draw synoptic 
ax1 = plt.subplot(gs[0,:2], projection=smap)
# smap.plot(axes=ax1, vmin=-100, vmax=100)
smap.plot(axes=ax1)

# plot single decay index 
ax2 = plt.subplot(gs[1, 0])
ax2.set_title("Decay Index at each point")
ax2.set_ylim([0, 4.5])
ax2.set_xlabel("height to solar surface [Mm]")
ax2.set_ylabel("decay index")
ax2.grid()

# plot averageed decay index
ax3 = plt.subplot(gs[1, 1])

# draw AIA and HMI contour
ax4 = plt.subplot(gs[:2,2:4], projection=cropped_amap)
cropped_amap.plot(axes=ax4, clip_interval=(1, 99.99)*u.percent)
grid = cropped_amap.draw_grid()

# levels = [-1000, -100, 100, 1000] * u.Gauss
# levels = [-500, -100, 100, 500] * u.Gauss
levels = [-50, 50] * u.Gauss
cset = cropped_hmap.draw_contours(levels, axes=ax4, cmap="bwr", alpha=0.7)

# plot decay index
# ax4 = plt.subplot(gs[1,2])

plt.show()


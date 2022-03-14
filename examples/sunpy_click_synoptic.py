import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
# from astropy.wcs import WCS

import sunpy
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE, HMI_LOS_IMAGE
from sunpy.physics.differential_rotation import solar_rotate_coordinate

import pfsspy
import pfsspy.utils

# newest reprojection function works with sunpy >=3.1, but this version
# has some problem to load old synoptic map
# from reproject_and_overlay import reproject_and_overlay

# button click and put marker function
def onclick(event):
    # get clicked pixel coordinates
    x = event.xdata
    y = event.ydata
    print('click: xdata=%f, ydata=%f' % (x, y))
    
    # transform the pixel to world (solar surface) coordinates
    coord = suba_s.pixel_to_world(x*u.pix, y*u.pix)
    
    # calibrate the difference of observed time
    # I should solve some warnings about the difference between "solar time" "Earth time"?
    coord_heligra = coord.transform_to(sunpy.coordinates.HeliographicCarrington)
    coord_syn = solar_rotate_coordinate(coord_heligra, time=smap.date)
    ax2.plot_coord(coord, color="white", marker="+", linewidth=5, markersize=10)
    ax1.plot_coord(coord_syn, color="white", marker="+", linewidth=5, markersize=10)
    plt.draw()

# data load
hmap = sunpy.map.Map(HMI_LOS_IMAGE).rotate(angle=180.0*u.deg)
amap = sunpy.map.Map(AIA_171_IMAGE)

# synoptic_sample = "../data/hmi.Synoptic_Mr.2111.fits"
# synoptic_sample = "../data/hmi.Synoptic_Mr_small.2111.fits" # smaller than the original

# if you want to download synoptic file at the specific date,
# execute following and it can download the file
from sunpy.coordinates.sun import carrington_rotation_number
from astropy.utils.data import download_file
car_num = carrington_rotation_number(amap.date)
# synoptic_url = f"http://jsoc.stanford.edu/data/hmi/synoptic/hmi.Synoptic_Mr.{int(car_num)}.fits"
synoptic_url = f"http://jsoc.stanford.edu/data/hmi/synoptic/hmi.Synoptic_Mr_small.{int(car_num)}.fits"
synoptic_sample = download_file(synoptic_url, cache=False)

smap = sunpy.map.Map(synoptic_sample)
# smap = reproject_and_overlay(synoptic_sample, HMI_LOS_IMAGE)

# the document says "if you have sunpy > 2.1 installed, this function is not needed"
# but I got some error without following procedure...
pfsspy.utils.fix_hmi_meta(smap)

# prepare matplotlib figure and connect click function 
fig = plt.figure(figsize=(20, 8))
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# cutout coordinates for AIA and HMI
blc_hs = SkyCoord(-200*u.arcsec, 0*u.arcsec, frame=hmap.coordinate_frame)
trc_hs = SkyCoord((-200+500)*u.arcsec, (0+500)*u.arcsec, frame=hmap.coordinate_frame)
blc_as = SkyCoord(-200*u.arcsec, 0*u.arcsec, frame=amap.coordinate_frame)
trc_as = SkyCoord((-200+500)*u.arcsec, (0+500)*u.arcsec, frame=amap.coordinate_frame)

# cutout AIA and HMI
subh_s = hmap.submap(blc_hs, top_right = trc_hs)
suba_s = amap.submap(blc_as, top_right = trc_as)

# draw synoptic 
ax1 = plt.subplot(1,2,1, projection=smap)
smap.plot(axes=ax1, vmin=-100, vmax=100)

# draw AIA and HMI contour
ax2 = plt.subplot(1,2,2, projection=suba_s)
suba_s.plot(axes=ax2, clip_interval=(1, 99.99)*u.percent)
grid = suba_s.draw_grid()

# levels = [-1000, -100, 100, 1000] * u.Gauss
# levels = [-500, -100, 100, 500] * u.Gauss
levels = [-50, 50] * u.Gauss
cset = subh_s.draw_contours(levels, axes=ax2, cmap="bwr", alpha=0.7)

plt.colorbar(cset,
             label=f"Magnetic Field Strength [{subh_s.unit}]",
             ticks=list(levels.value) + [0],
             shrink=0.7,
             pad=0.17)
plt.show()


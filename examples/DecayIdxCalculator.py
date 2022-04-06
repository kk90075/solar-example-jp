import datetime
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
from scipy import interpolate

import astropy
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import sunpy
import sunpy.map
from sunpy.map.sources.sdo import HMIMap #, HMISynopticMap
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import Helioprojective, RotatedSunFrame, transform_with_sun_center, propagate_with_solar_surface
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

class DecayIdxCalculator:
    def __init__(self, nr, rss):
        self.jsoc_email = JSOC_EMAIL
        self.nr = nr
        self.rss = rss
        self.effects = [withStroke(linewidth=2, foreground="white")]

    
    def _fido_get_aia(self, wl_aia, t_aia):
        self.t_aia = astropy.time.Time(t_aia, scale='utc')
        _q_aia = Fido.search(
            a.Time(self.t_aia, self.t_aia + 5 * u.min),
            a.Wavelength(wl_aia * u.angstrom),
            a.Instrument.aia,
            # a.jsoc.Notify(JSOC_EMAIL),
        )
        _d_aia = Fido.fetch(_q_aia[0][0])
        if self.showfilename:
            print(_q_aia[0][0])
        return sunpy.map.Map(_d_aia)

    def _fido_get_hmi(self, t_hmi):
        _t_hmi = astropy.time.Time(t_hmi, scale='utc')
        _q_hmi = Fido.search(
            a.Time(_t_hmi, _t_hmi + 5 * u.min),
            a.Instrument.hmi,
            a.Physobs.los_magnetic_field,
            # a.jsoc.Notify(JSOC_EMAIL),
        )
        _d_hmi = Fido.fetch(_q_hmi[0][0])
        if self.showfilename:
            print(_q_hmi[0][0])
        return sunpy.map.Map(_d_hmi)

    def _fido_get_synoptic(self):
        # _t_syn = astropy.time.Time(_t_syn, scale='utc')
        _car_num = carrington_rotation_number(self.t_aia)
        _q_syn = Fido.search(
            a.Time(self.t_aia, self.t_aia),
            a.jsoc.Series('hmi.synoptic_mr_polfil_720s'),
            a.jsoc.PrimeKey('CAR_ROT', int(_car_num)),
            a.jsoc.Notify(JSOC_EMAIL),
        )
        _d_syn = Fido.fetch(_q_syn[0][0])
        if self.showfilename:
            print(_d_syn[0][0])
        return sunpy.map.Map(_d_syn)

    def set_fido_file(self, wl_aia, t_aia, t_hmi, downsample=True, showfilename=False):
        self.showfilename = showfilename
        self.map_downsample = downsample
        self.aiamap = self._fido_get_aia(wl_aia, t_aia)
        self.hmimap = self._fido_get_hmi(t_hmi)
        self.synmap = self._fido_get_synoptic()
        self._reproject_maps()

    def set_local_file(self, amap, hmap, smap, downsample=True):
        self.map_downsample = downsample
        self.aiamap = amap
        self.hmimap = hmap
        self.synmap = smap
        self._reproject_maps()

    def _reproject_maps(self): # TODO overlay hmi batch to synoptic
        if self.map_downsample:
            self.new_aiamap = self.aiamap.resample([2048, 2048]*u.pix)
        else:
            self.new_aiamap = self.aiamap
        self.new_hmimap = self._hmi_to_aia()
        self.new_synmap = self._overlay_hmi_to_syn()

    # TODO
    # the reprojection is almost done, but it takes ~1 minutes when you use
    # full resolution AIA/HMI map, so it would be better for the reprojection
    # is applied only for cropped map ?
    def _hmi_to_aia(self):
        if self.map_downsample:
            _hmap = self.hmimap.resample([2048, 2048]*u.pix)
        else:
            _hmap = self.hmimap
        out_frame = Helioprojective(observer='earth', obstime=self.new_aiamap.date)
        out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
        header = sunpy.map.make_fitswcs_header(self.new_aiamap.data.shape,
                                            out_center,
                                            scale=u.Quantity(_hmap.scale))
        out_wcs = WCS(header)
        with propagate_with_solar_surface():
            warped_hmap = _hmap.reproject_to(out_wcs)
        newmap = HMIMap(warped_hmap.data, warped_hmap.meta)
        newmap.meta['bunit'] = 'Gauss' # If it is 
        return newmap

    # TODO
    def _overlay_hmi_to_syn(self):
        return self.synmap.resample([720, 360] * u.pix)

    def _onclick_preplot(self, event):
        # get clicked pixel coordinates
        _x = event.xdata
        _y = event.ydata
        print('click: xdata=%f, ydata=%f' % (_x, _y))
        # transform the pixel to world (solar surface) coordinates
        self.center_coord = self.new_aiamap.pixel_to_world(_x*u.pix, _y*u.pix)

    # button click and put marker function
    def _onclick(self, event):
        # get clicked pixel coordinates
        _x = event.xdata
        _y = event.ydata
        print('click: xdata=%f, ydata=%f' % (_x, _y))
        
        # transform the pixel to world (solar surface) coordinates
        coord = self.cropped_aiamap.pixel_to_world(_x*u.pix, _y*u.pix)
        
        # calibrate the difference of observed time
        # I should solve some warnings about the difference between "solar time" "Earth time"?
        coord_heligra = coord.transform_to(sunpy.coordinates.HeliographicCarrington)
        coord_syn = solar_rotate_coordinate(coord_heligra, time=self.new_synmap.date)
        c_spix = self.new_synmap.world_to_pixel(coord_syn)

        # with open(f"coord_{dt_str}.txt", mode="a+") as f:
        #     f.writelines(f"{c_spix[0].value},{c_spix[1].value}\n")

        self.ax_click.plot_coord(coord, marker="+", linewidth=10, markersize=12, path_effects=self.effects)
        self.ax_syn.plot_coord(coord_syn, color="white", marker="+", linewidth=5, markersize=10)

        di_click_point = self.click_point_decay(c_spix.x.value, c_spix.y.value)
        # di_one, h_one = self.interp_decay(di, , height, h_limit)
        for i in range(di_click_point.shape[0]):
            for j in range(di_click_point.shape[1]):
                self.decay_index_list.append(di_click_point[i,j])
        self.ax_eachdi.plot(self.h_interp, self.interp_decay(np.average(di_click_point, axis=(0,1))))

        di_ave = np.average(np.array(self.decay_index_list), axis=0)
        di_std = np.std(np.array(self.decay_index_list), axis=0)
        self.ax_avedi.cla()
        self.ax_avedi.plot(self.h_interp, self.interp_decay(di_ave), color="black")
        self.ax_avedi.errorbar(self.h_limited, di_ave, yerr=di_std, fmt="none", ecolor="black", elinewidth=1)
        self.ax_avedi.set_ylim([0, 4.5])
        self.ax_avedi.set_title("Averaged Decay Index")
        self.ax_avedi.set_xlabel("height to solar surface [Mm]")
        self.ax_avedi.set_ylabel("decay index")
        self.ax_avedi.grid()
        plt.draw()

    def cal_decay(self, h_threshold):
        # 'h_threshold' is the maximal height to interpolate (unit: Mm)
        self.h_threshold = h_threshold
        self.decay_index_list = []
        pfss_in = pfsspy.Input(self.new_synmap, self.nr, self.rss)
        self.pfss_out = pfsspy.pfss(pfss_in)
        
        b_theta = self.pfss_out.bg[:,:,:,1]
        b_phi = self.pfss_out.bg[:,:,:,0]
        # b_r = pfss_out.bg[:,:,:,2]
        bh = np.sqrt(b_theta*b_theta + b_phi*b_phi)
        h = (np.exp(self.pfss_out.grid.rg) - 1)[1:]

        dln_bh = np.diff(np.log(bh[:,:,1:]), axis=-1)
        dln_h = np.diff(np.log(h))
        di = -dln_bh/dln_h
        # transpose axis same as synoptic map
        # di = di.transpose(1,0,2)
        # assume the index at solar surface to be same as the index one step above
        di = np.pad(di, [(0,0),(0,0),(1,0)], "edge")
        
        self.decay_index = di
        # self.calc_height = h
        self.height_Mm = (np.exp(self.pfss_out.grid.rg) - 1) * const.R_sun.to("Mm").value

        self.h_limited = self.height_Mm[np.where(self.height_Mm<=self.h_threshold)[0]]
        self.h_interp = np.linspace(0, math.floor(self.h_limited[-1]), 1000)

        fig = plt.figure(figsize=(16, 8))
        gs = matplotlib.gridspec.GridSpec(2,4)
        cid = fig.canvas.mpl_connect('button_press_event', self._onclick)

        # draw synoptic 
        self.ax_syn = plt.subplot(gs[0,:2], projection=self.new_synmap)
        self.new_synmap.plot(axes=self.ax_syn)

        # plot each decay index 
        self.ax_eachdi = plt.subplot(gs[1, 0])
        self.ax_eachdi.set_title("Decay Index at each point")
        self.ax_eachdi.set_ylim([0, 4.5])
        self.ax_eachdi.set_xlabel("height to solar surface [Mm]")
        self.ax_eachdi.set_ylabel("decay index")
        self.ax_eachdi.grid()

        # plot averageed decay index
        self.ax_avedi = plt.subplot(gs[1, 1])

        # draw AIA and HMI contour
        self.ax_click = plt.subplot(gs[:2,2:4], projection=self.cropped_aiamap)
        self.cropped_aiamap.plot(axes=self.ax_click, clip_interval=(1, 99.99)*u.percent)
        grid = self.cropped_aiamap.draw_grid()

        # levels = [-1000, -100, 100, 1000] * u.Gauss
        # levels = [-500, -100, 100, 500] * u.Gauss
        levels = [-50, 50] * u.Gauss
        cset = self.cropped_hmimap.draw_contours(levels, axes=self.ax_click, cmap="bwr", alpha=0.7)

        plt.show()

    def click_point_decay(self, x, y): # > onclick ?
        # get decay indecies at the nearest 4 grid points 
        di_limited = self.decay_index[:,:,np.where(self.height_Mm <= self.h_threshold)[0]]
        # di_averaged = np.average(di_limited[math.floor(y):math.floor(y)+2,math.floor(x):math.floor(x)+2], axis=(0,1))
        return di_limited[math.floor(x):math.floor(x)+2,math.floor(y):math.floor(y)+2]

    def interp_decay(self, di_in_question):
        f = interpolate.interp1d(self.h_limited, di_in_question, kind="cubic")
        di_interp = f(self.h_interp)
        return di_interp

    def preplot(self):
        fig_pre = plt.figure(figsize=(9,9))
        ax_pre = fig_pre.add_subplot(1,1,1, projection=self.new_aiamap)
        cid = fig_pre.canvas.mpl_connect('button_press_event', self._onclick_preplot)
        self.new_aiamap.plot(axes = ax_pre, clip_interval = (1, 99.99)*u.percent)
        # levels = [-50, 50] * u.Gauss
        # cset = self.new_hmimap.draw_contours(levels, axes=ax_pre, cmap="bwr", alpha=0.7)
        plt.show()
        plt.close()

        c_lon = int(self.center_coord.Tx.value)
        c_lat = int(self.center_coord.Ty.value)

        print(f"last clicked coordinates are x:{c_lon} y:{c_lat}")
        print("please input the height of cropping (unit: arcsec)")
        height = int(input())
        print("please input the width of cropping (unit: arcsec)")
        width = int(input())
        # height, width = 500, 500 # DEBUG

        fig_crop = plt.figure(figsize=(9,9))
        self.blc_a = SkyCoord(Tx=(c_lon-int(width/2))*u.arcsec,Ty=(c_lat-int(height/2))*u.arcsec,frame=self.new_aiamap.coordinate_frame)
        self.trc_a = SkyCoord(Tx=(c_lon+int(width/2))*u.arcsec,Ty=(c_lat+int(height/2))*u.arcsec,frame=self.new_aiamap.coordinate_frame)
        self.blc_h = SkyCoord(Tx=(c_lon-int(width/2))*u.arcsec,Ty=(c_lat-int(height/2))*u.arcsec,frame=self.new_hmimap.coordinate_frame)
        self.trc_h = SkyCoord(Tx=(c_lon+int(width/2))*u.arcsec,Ty=(c_lat+int(height/2))*u.arcsec,frame=self.new_hmimap.coordinate_frame)
        # print(blc, trc)
        self.cropped_aiamap = self.new_aiamap.submap(self.blc_a, top_right=self.trc_a)
        self.cropped_hmimap = self.new_hmimap.submap(self.blc_h, top_right=self.trc_h)
        ax_crop = fig_crop.add_subplot(1,1,1, projection = self.cropped_aiamap)
        self.cropped_aiamap.plot()
        levels = [-50, 50] * u.Gauss
        cset = self.cropped_hmimap.draw_contours(levels, axes=ax_crop, cmap="bwr", alpha=0.7)
        plt.show()

    def select_coordinates(self):
        exit = True
        while exit:
            self.preplot()
            print("if you finish and go next, input 'y' (if repeat, input some other key):")
            pressed = str(input()).lower()
            if pressed == "y":
                exit = False


    
if __name__ == '__main__':
    # t_aia = datetime.datetime(2014, 10, 22, 14, 27) # (2012, 3, 14, 15, 21)
    # t_aia = datetime.datetime(2017, 7, 14, 1, 1) # 191
    # t_aia = datetime.datetime(2014, 4, 18, 12, 55) # 161
    t_aia = datetime.datetime(2011, 8, 4, 3, 32) # 024
    t_hmi = t_aia
    nr, rss = 100, 2
    h_limit = 200 # Mm
    key_di = 1.5

    hmap = sunpy.map.Map("/Users/kihara/sunpy/data/hmi_m_45s_2017_07_14_01_01_30_tai_magnetogram.fits")
    amap = sunpy.map.Map("/Users/kihara/sunpy/data/aia_lev1_1600a_2017_07_14t01_03_50_12z_image_lev1.fits")
    smap = sunpy.map.Map("/Users/kihara/sunpy/data/hmi.synoptic_mr_polfil_720s.2192.Mr_polfil.fits")

    DIC = DecayIdxCalculator(nr, rss)
    # DIC.set_fido_file(1600, t_aia, t_hmi, downsample=True, showfilename=True)
    DIC.set_local_file(amap, hmap, smap, downsample=True)
    DIC.select_coordinates()
    DIC.cal_decay(h_limit)


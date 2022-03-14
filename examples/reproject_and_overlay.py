import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const

import sunpy
import sunpy.map

from reproject import reproject_interp
from sunpy.coordinates import propagate_with_solar_surface

import copy

import pfsspy
import pfsspy.utils

# reproject some hmi images to synoptic coordinates and 
# overlay on the existing synoptic map
# this function uses only -60 ~ 60 degrees both in longitudes and latitude
# if you want to use another area, please change below the "generate mask for ..."
def reproject_and_overlay(synoptic_path, hmi_path):
    synmap = sunpy.map.Map(synoptic_path)
    pfsspy.utils.fix_hmi_meta(synmap)

    hmimap = sunpy.map.Map(hmi_path)

    # generate mask for 60 degree lon/lat
    # https://docs.sunpy.org/en/stable/generated/gallery/map/map_segment.html#sphx-glr-generated-gallery-map-map-segment-py
    all_hpc = sunpy.map.all_coordinates_from_map(hmimap)
    all_hgs = all_hpc.transform_to("heliographic_stonyhurst")
    segment_mask = np.logical_or(all_hgs.lon >= 60 * u.deg, all_hgs.lon <= -60 * u.deg)
    segment_mask = np.logical_or(all_hgs.lat >= 60 * u.deg, all_hgs.lat <= -60 * u.deg)
    segment_mask |= np.isnan(all_hgs.lon)
    segment_mask |= np.isnan(all_hgs.lat)

    # prepare segmented hmi
    segmented_data = copy.deepcopy(hmimap.data)
    segmented_data[np.where(segment_mask==True)] = np.nan
    segmented_map = sunpy.map.Map(segmented_data, hmimap.meta)

    # reproject hmi / sunpy >= 3.1 is needed
    # if you do not use "propaget_with_solar_surface", the differential rotation
    # is not considered and reproject is not correct
    # https://docs.sunpy.org/en/stable/generated/gallery/differential_rotation/reprojected_map.html
    with propagate_with_solar_surface():
        reprojected_data, footprint = reproject_interp(segmented_map, synmap.wcs, synmap.data.shape)

    # overlay the reprojected/segmented hmi
    repro_and_seg = copy.deepcopy(synmap.data)
    repro_and_seg = np.where(np.isnan(reprojected_data), repro_and_seg, reprojected_data)
    repro_and_seg = np.where(np.isnan(repro_and_seg), 0, repro_and_seg) # replace np.nan in the raw synoptic

    newmap = sunpy.map.Map(repro_and_seg, synmap.meta)
    pfsspy.utils.fix_hmi_meta(newmap)

    return newmap

if __name__=="__main__":
    synoptic_2192 = "/Users/kihara/Files/research/02_SEP_CDAW/SEP_D/211113/calc_pfss/hmi.Synoptic_Mr_small.2192.fits"
    hmi_file = "/Users/kihara/Files/research/02_SEP_CDAW/SEP_D/data/191_20170714/hmi/hmi.m_45s.20170714_000000_TAI.2.magnetogram.fits"

    smap = sunpy.map.Map(synoptic_2192)
    hmap = sunpy.map.Map(hmi_file)

    car_rot = smap.meta["car_rot"]
    proj_date = hmap.meta["t_obs"]
    newtitle = f"HMI synoptic {car_rot} + magnetogram at {proj_date}"

    c_smap = reproject_and_overlay(synoptic_2192, hmi_file)
    
    fig = plt.figure(figsize=(8,7))
    ax1 = fig.add_subplot(2,1,1, projection=smap)
    ax2 = fig.add_subplot(2,1,2, projection=c_smap)
    smap.plot(axes=ax1)
    c_smap.plot(axes=ax2, title=newtitle)
    plt.show()
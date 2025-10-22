#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cmath

from mpl_toolkits.basemap import Basemap
from datetime import datetime
from datetime import timedelta
from scipy.interpolate import griddata
from math import *




def isodepth_function(m):
    
    # Slab isodepth curves

    isodepth_lines         = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/Japan_slab2_dep_02.24.18.xyz", header=None, delimiter='\t')
    isodepth_lines.columns = ['Lon','Lat','Depth' ]
    isodepth_lines         = isodepth_lines.dropna()
    isodepth_lines         = isodepth_lines.astype(float)
    isodepth_lines['Lon']  = isodepth_lines['Lon']

    isodepth_lines20 = isodepth_lines[isodepth_lines["Depth"] == -20]
    isodepth_lines20 = isodepth_lines20.set_index([np.arange(0, len(isodepth_lines20))])

    isodepth_lines40 = isodepth_lines[isodepth_lines["Depth"] == -40]
    isodepth_lines40 = isodepth_lines40.set_index([np.arange(0, len(isodepth_lines40))])

    isodepth_lines60 = isodepth_lines[isodepth_lines["Depth"] == -60]
    isodepth_lines60 = isodepth_lines60.set_index([np.arange(0, len(isodepth_lines60))])

    isodepth_lines80 = isodepth_lines[isodepth_lines["Depth"] == -80]
    isodepth_lines80 = isodepth_lines80.set_index([np.arange(0, len(isodepth_lines80))])

    isodepth_lines100 = isodepth_lines[isodepth_lines["Depth"] == -100]
    isodepth_lines100 = isodepth_lines100.set_index([np.arange(0, len(isodepth_lines100))])


    # Trench
    trench = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/all_segment", header=None, names =["Lon", "Lat"], delimiter=' ')
    trench = trench[(35 - 1 <= trench["Lat"]) & (trench["Lat"] <= 40.5 + 1)]
    trench = trench.set_index([np.arange(0, len(trench))])


    # isodepth lines
    for iso in [isodepth_lines20, isodepth_lines40, isodepth_lines60, isodepth_lines80, isodepth_lines100]:
        map_lat_isodepth = []
        map_lon_isodepth = []
        for i in range(len(iso)):
            x, y = m(iso['Lon'][i], iso['Lat'][i])
            map_lon_isodepth.append(x)
            map_lat_isodepth.append(y)
        plt.plot(map_lon_isodepth, map_lat_isodepth, color='black', zorder = 2)

    # trench line
    map_lat_trench = []
    map_lon_trench = []
    for i in range(len(trench)):
        x, y = m(trench['Lon'][i], trench['Lat'][i])
        map_lon_trench.append(x)
        map_lat_trench.append(y)
    #plt.plot(map_lon_trench, map_lat_trench, color='black', zorder = 2)
    plt.plot(map_lon_trench, map_lat_trench, marker = 8, color='black', zorder = 2)


    """
    # Slab
    map_lat_slab = []
    map_lon_slab = []
    for i in range(len(slab_japan_up)):
        x, y = m(slab_japan_up['Lon'][i], slab_japan_up['Lat'][i])
        map_lon_slab.append(x)
        map_lat_slab.append(y)
    m.scatter(map_lon_slab, map_lat_slab, c = slab_japan_up["Depth"], cmap=plt.cm.gist_earth, alpha = 0.2, zorder = 2)
    """
    
    
    
"""   
def add_slab_depth(catalog_japan):
    #### Slab surface Hayes 2018

    # Upper slab boundary : Hayes
    slab_japan_up = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Japan_slab2_dep_02.24.18.xyz", names = ["Lon", "Lat", "Depth"])
    slab_japan_up = slab_japan_up.dropna()
    slab_japan_up = slab_japan_up.set_index([np.arange(0, len(slab_japan_up))])

    # Interpolation to have slab depth at each EQ position (lon-lat)
    lon_to_find                 = catalog_japan["Lon"]
    lat_to_find                 = catalog_japan["Lat"]
    depth_interpolation         = griddata((slab_japan_up["Lon"], slab_japan_up["Lat"]), slab_japan_up["Depth"], (lon_to_find, lat_to_find), method='linear')
    catalog_japan["Slab Depth"] = np.abs(depth_interpolation)

    #print("Lon min slab : ", slab_japan_up["Lon"].min(), "/ Lon max slab :", slab_japan_up["Lon"].max())
    #print("Lat min slab : ", slab_japan_up["Lat"].min(), "/ Lat max slab : ",  slab_japan_up["Lat"].max())
    #Distance to top slab negative when under the slab surface

    catalog_japan["Distance to top slab"] = catalog_japan["Slab Depth"] - catalog_japan["Depth"]
    return(catalog_japan)
"""
    

    
def find_tohoku(catalog_japan):
    return(catalog_japan[(datetime(2011,3,11) <= catalog_japan["Date_UTC"]) & (catalog_japan["Date_UTC"] < datetime(2011,3,12)) & (catalog_japan["Mag"] >= 9)])
    #print("'tohoku' :")
    #display(tohoku)
    #return(tohoku)
    
    
    
    
def selection_catalog(catalog_japan, t_min, t_max , mc, limit_depth_deep, limit_depth_shallow, lat_min, lat_max, lon_min, lon_min_shallow, lon_max):
    # New column in catalog japan containing the distance from Tohoku epicenter
    # catalog_japan["Dist_Tohoku"] = np.sqrt(((catalog_japan.Lat.values - tohoku.Lat.values)*111)**2 + ((catalog_japan.Lon.values - tohoku.Lon.values)*111*np.cos(np.radians(tohoku.Lat.values)))**2)

    

    # Define the interest area
    #lat_min         = 37  
    #lat_max         = 40   
    #lon_min         = 140
    #lon_min_shallow = lon_min # to change area selected for shallow EQs
    #lon_max         = 144


    # Conditions
    c_latmin          = (catalog_japan.Lat.values>lat_min)
    c_latmax          = (catalog_japan.Lat.values<lat_max)
    c_lonmin          = (catalog_japan.Lon.values>lon_min)
    c_lonmin_shallow  = (catalog_japan.Lon.values>lon_min_shallow) 
    c_lonmax          = (catalog_japan.Lon.values<lon_max)
    #c_dist            = (catalog_japan["Dist_Tohoku"] < 75)
    c_tmax            = (catalog_japan["Date_UTC"]<(t_max))
    c_tmin            = (catalog_japan["Date_UTC"]>(t_min))
    c_mag             = (catalog_japan.Mag >= mc)
    c_deep            = ((catalog_japan.Depth >= limit_depth_deep) & (catalog_japan['Distance to top slab'] <= 0)) #, under slab surface 
    c_shallow         = ((catalog_japan.Depth < limit_depth_shallow) & (catalog_japan['Distance to top slab']<= 5))
    c_between         = ((catalog_japan.Depth < limit_depth_deep) & (catalog_japan.Depth >= limit_depth_shallow)  & (catalog_japan['Distance to top slab']<= 0))

    # Area defined with lat lon values
    EQ_deep      = catalog_japan[c_latmin & c_latmax & c_lonmin         & c_lonmax & c_tmax & c_tmin & c_mag & c_deep]
    EQ_shallow   = catalog_japan[c_latmin & c_latmax & c_lonmin_shallow & c_lonmax & c_tmax & c_tmin & c_mag & c_shallow]
    EQ_between   = catalog_japan[c_latmin & c_latmax & c_lonmin_shallow & c_lonmax & c_tmax & c_tmin & c_mag & c_between]

    # Zone plus large pour compter les M>=6
    large_extent = 4 #degres to add to extend the area
    c_latmin_large          = (catalog_japan.Lat.values>lat_min-large_extent)
    c_latmax_large          = (catalog_japan.Lat.values<lat_max+large_extent)
    c_lonmin_large          = (catalog_japan.Lon.values>lon_min-large_extent)
    c_lonmin_shallow_large  = (catalog_japan.Lon.values>lon_min_shallow-large_extent) 
    c_lonmax_large          = (catalog_japan.Lon.values<lon_max+large_extent)

    EQ_deep_large      = catalog_japan[c_latmin_large & c_latmax_large & c_lonmin_large         & c_lonmax_large & c_tmax & c_tmin & c_mag & c_deep]
    EQ_shallow_large   = catalog_japan[c_latmin_large & c_latmax_large & c_lonmin_shallow_large & c_lonmax_large & c_tmax & c_tmin & c_mag & c_shallow]
    EQ_between_large   = catalog_japan[c_latmin_large & c_latmax_large & c_lonmin_shallow_large & c_lonmax_large & c_tmax & c_tmin & c_mag & c_between]
    EQ_all_large       = catalog_japan[c_latmin_large & c_latmax_large & c_lonmin_large & c_lonmax_large & c_tmax & c_tmin & c_mag]



    #print("Number of shallow events (EQ_shallow): ", len(EQ_shallow))
    #print("Number of deep events (EQ_deep): ", len(EQ_deep))
    #print("Number of intermediate events (EQ_between): "   , len(EQ_between))
    return(EQ_deep, EQ_shallow, EQ_between, EQ_deep_large, EQ_shallow_large, EQ_between_large, EQ_all_large)


def count_events(catalog, start, end, m_min):
    if m_min == None:
        return( len(catalog[(catalog['Date_UTC'] >= start) & (catalog['Date_UTC'] <= end)]))
    else :
        return( len(catalog[(catalog['Date_UTC'] >= start) & (catalog['Date_UTC'] <= end) & (catalog['Mag'] >= m_min)]) )





def add_slab_depth_kita(catalog_japan):
    # Read Slab 2
    slab_japan_up = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/Japan_slab2_dep_02.24.18.xyz", names = ["Lon", "Lat", "Depth"])
    slab_japan_up = slab_japan_up.dropna()
    slab_japan_up = slab_japan_up.set_index([np.arange(0, len(slab_japan_up))])
    #slab_japan_up = slab_japan_up[(slab_japan_up["Lon"] >= lon_min ) & (slab_japan_up["Lon"]<= lon_max) & (slab_japan_up["Lat"]>=lat_min) & (slab_japan_up["Lat"]<=lat_max)]
    #slab_japan_up

    # Upper slab boundary : Kita
    slab_kita = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/plate_data/PAC/plate_combine.dat", names = ["Lon", "Lat", "Depth"], sep='\t')
    slab_kita["Depth"] = - slab_kita["Depth"]
    slab_kita = slab_kita.dropna()
    slab_kita = slab_kita.set_index([np.arange(0, len(slab_kita))])
    #slab_kita = slab_kita[(slab_kita["Lon"] >= lon_min ) & (slab_kita["Lon"]<= lon_max) & (slab_kita["Lat"]>=lat_min) & (slab_kita["Lat"]<=lat_max)]
    #display(slab_kita)


    # Interpolation to have slab depth at each EQ position of slab 2
    lon_to_find = slab_japan_up["Lon"]
    lat_to_find = slab_japan_up["Lat"]
    depth_interpolation_kita = griddata((slab_kita["Lon"], slab_kita["Lat"]), slab_kita["Depth"], (lon_to_find, lat_to_find), method='linear')


    slab_kita_interpolated = pd.DataFrame()
    slab_kita_interpolated["Depth"] = np.abs(depth_interpolation_kita)
    slab_kita_interpolated["Lon"]   = slab_japan_up["Lon"].values
    slab_kita_interpolated["Lat"]   = slab_japan_up["Lat"].values



    # Distance to top slab of earthquakes
    # Interpolation to have slab depth at each EQ position (lon-lat)
    lon_to_find_cat                 = catalog_japan["Lon"]
    lat_to_find_cat                  = catalog_japan["Lat"]
    depth_interpolation_cat         = griddata((slab_kita_interpolated["Lon"],slab_kita_interpolated["Lat"]), slab_kita_interpolated["Depth"], (lon_to_find_cat , lat_to_find_cat ), method='linear')
    catalog_japan["Slab Depth"] = np.abs(depth_interpolation_cat)

    #print("Lon min slab : ", slab_japan_up["Lon"].min(), "/ Lon max slab :", slab_japan_up["Lon"].max())
    #print("Lat min slab : ", slab_japan_up["Lat"].min(), "/ Lat max slab : ",  slab_japan_up["Lat"].max())
    #Distance to top slab negative when under (ABOVE ???) the slab surface

    catalog_japan["Distance to top slab"] = catalog_japan["Slab Depth"] - catalog_japan["Depth"]
    return(catalog_japan)

# In[2]:


""" 
EQ_shallow_large[" Flag "] = 1
EQ_deep_large[" Flag "]    = 2
EQ_selection               = pd.concat((EQ_shallow_large, EQ_deep_large)).sort_values(by = "Date UTC")
EQ_selection_2             = pd.DataFrame(columns=['Date UTC', 'Flag'])
EQ_selection_2["Date UTC"] = EQ_selection["Date UTC"]
EQ_selection_2["Flag"]     = EQ_selection[" Flag "]
EQ_selection_2
EQ_selection_2.to_csv("/Users/choulia/Documents/Doctorat/catalogues/Japan/EQ_selection_2.csv", index=None)
"""


# In[ ]:


#Projection orthogonale pour prendre section perpendiculaire à la trench
def proj_ortho(lon_start_point,lat_start_point,slope_angle,all_lon_events,all_lat_events):
    from pyproj import Proj

    # Pour passer de geo à cartésien
    p = Proj(proj='utm',zone = 54, south = False, ellps='WGS84', preserve_units=False)
    # Un point sur la trench qui sera notre référence, 0m
    x_start_point, y_start_point = p(lon_start_point, lat_start_point)
    point_start_section = 0
    slope          = np.cos(np.radians(slope_angle))

    all_dist = np.zeros(len(all_lat_events))
    all_x_proj_section = np.zeros(len(all_lat_events))

    for i in np.arange(len(all_lat_events)):
        lon_event = all_lon_events[i]
        lat_event = all_lat_events[i]
        x_event,y_event = p(lon_event, lat_event)
        x_proj = (-y_start_point*slope+x_start_point*slope**2+y_event*slope+x_event)/(1+slope**2)
        y_proj = slope*x_proj+(y_start_point-slope*x_start_point)
        dx = x_proj - x_start_point
        dy = y_proj - y_start_point

        x_proj_section = point_start_section + np.sqrt(dx**2+dy**2)
        dist = np.sqrt((x_event-x_proj)**2+(y_event-y_proj)**2) #hypothénuse = distance

        all_dist[i] = dist
        all_x_proj_section[i] = x_proj_section
    return(all_dist, all_x_proj_section)




def map_background(lon_min, lon_max, lat_min, lat_max):
    m = Basemap(projection="merc", llcrnrlat=lat_min, urcrnrlat=lat_max,
            llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='f')  # mercator
    m.fillcontinents(alpha=0.9)
    m.drawmapboundary(fill_color='azure')
    m.drawparallels(np.arange(lat_min, lat_max, 1), labels=[
                1, 0, 0, 1], color='grey', fontsize=24)
    m.drawmeridians(np.arange(lon_min, lon_max, 1), labels=[
                1, 1, 0, 1], rotation=45, color='grey', fontsize=24)
    return m


def plot_trench(m, lon_min, lon_max, lat_min, lat_max):
    # Trench line importation
    # Trench
    trench = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/all_segment", header=None, names =["Lon", "Lat"], delimiter=' ')
    trench = trench[(35 - 1 <= trench["Lat"]) & (trench["Lat"] <= 41)]
    trench = trench[(139 <= trench["Lon"]) & (trench["Lon"] <= 144 + 1)]
    trench = trench.sort_values(by=['Lon'])
    trench = trench.set_index([np.arange(0, len(trench))])
    
    
    # trench line
    map_lat_trench = []
    map_lon_trench = []
    for i in range(len(trench)):
        x, y = m(trench['Lon'][i], trench['Lat'][i])
        map_lon_trench.append(x)
        map_lat_trench.append(y)
    #plt.plot(map_lon_trench, map_lat_trench, color='black', zorder = 2)
    plt.plot(map_lon_trench, map_lat_trench, marker = 8, color='black', zorder = 2, markersize=9 )

def interpolate_slab_kita(lon_min, lon_max, lat_min, lat_max):
    # Read Slab 2
    slab_japan_up = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/Japan_slab2_dep_02.24.18.xyz", names = ["Lon", "Lat", "Depth"])

    slab_japan_up = slab_japan_up.dropna()
    slab_japan_up = slab_japan_up.set_index([np.arange(0, len(slab_japan_up))])
    slab_japan_up = slab_japan_up[(slab_japan_up["Lon"] >= lon_min ) & (slab_japan_up["Lon"]<= lon_max) & (slab_japan_up["Lat"]>=lat_min) & (slab_japan_up["Lat"]<=lat_max)]
    slab_japan_up

    # Upper slab boundary : Kita
    slab_kita = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/plate_data/PAC/plate_combine.dat", names = ["Lon", "Lat", "Depth"], sep='\t')
    slab_kita["Depth"] = - slab_kita["Depth"]
    slab_kita = slab_kita.dropna()
    slab_kita = slab_kita.set_index([np.arange(0, len(slab_kita))])
    slab_kita = slab_kita[(slab_kita["Lon"] >= lon_min ) & (slab_kita["Lon"]<= lon_max) & (slab_kita["Lat"]>=lat_min) & (slab_kita["Lat"]<=lat_max)]
    display(slab_kita)


    # Interpolation to have slab depth at each EQ position of slab 2
    lon_to_find = slab_japan_up["Lon"]
    lat_to_find = slab_japan_up["Lat"]
    depth_interpolation_kita = griddata((slab_kita["Lon"], slab_kita["Lat"]), slab_kita["Depth"], (lon_to_find, lat_to_find), method='linear')


    slab_kita_interpolated = pd.DataFrame()
    slab_kita_interpolated["Depth"] = depth_interpolation_kita
    slab_kita_interpolated["Lon"]   = slab_japan_up["Lon"].values
    slab_kita_interpolated["Lat"]   = slab_japan_up["Lat"].values

    return(slab_kita_interpolated)
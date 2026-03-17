from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from datetime import timedelta
from scipy.interpolate import griddata
from math import *
import matplotlib.transforms as transforms
from ast import literal_eval
from matplotlib.patches import Ellipse



def add_slab_depth(catalog_japan):
    #### Slab surface Hayes 2018

    # Upper slab boundary : Hayes
    slab_japan_up = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/Japan_slab2_dep_02.24.18.xyz", names = ["Lon", "Lat", "Depth"])
    slab_japan_up = slab_japan_up.dropna()
    slab_japan_up = slab_japan_up.set_index([np.arange(0, len(slab_japan_up))])

    # Interpolation to have slab depth at each EQ position (lon-lat)
    lon_to_find                 = catalog_japan["Lon"]
    lat_to_find                 = catalog_japan["Lat"]
    depth_interpolation         = griddata((slab_japan_up["Lon"], slab_japan_up["Lat"]), slab_japan_up["Depth"], (lon_to_find, lat_to_find), method='linear')
    catalog_japan["Slab Depth"] = np.abs(depth_interpolation)

    catalog_japan["Distance to top slab"] = catalog_japan["Slab Depth"] - catalog_japan["Depth"]
    return(catalog_japan)


def map_background(lon_min, lon_max, lat_min, lat_max):
    m = Basemap(projection="merc", llcrnrlat=lat_min, urcrnrlat=lat_max,
            llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='h')  # mercator
    m.fillcontinents(alpha=0.9)
    m.drawmapboundary(fill_color='azure')
    m.drawparallels(np.arange(lat_min, lat_max, 1), labels=[
                1, 0, 0, 1], color='grey', fontsize=28)
    m.drawmeridians(np.arange(lon_min, lon_max, 1), labels=[
                1, 1, 0, 1], rotation=20, color='grey', fontsize=28)
    return m

def plot_isodepth_lines_japan(m):
    # isodepth lines importation, line by line
    isodepth_lines         = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab2_contours_japan.txt", header=None, delimiter='\t')
    isodepth_lines.columns = ['Lon','Lat','Depth' ]
    isodepth_lines         = isodepth_lines.dropna()
    isodepth_lines         = isodepth_lines.astype(float)
    isodepth_lines['Lon']  = isodepth_lines['Lon'] - 360
    
    #isodepth_lines = isodepth_lines[(35 - 1 <= isodepth_lines["Lat"]) & (isodepth_lines["Lat"] <= 41)]
    #isodepth_lines = isodepth_lines[(139 <= isodepth_lines["Lon"]) & (isodepth_lines["Lon"] <= 144 + 1)]
    
    isodepth_lines = isodepth_lines.sort_values(by=['Lon'])

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
    
    isodepth_lines120 = isodepth_lines[isodepth_lines["Depth"] == -120]
    isodepth_lines120 = isodepth_lines120.set_index([np.arange(0, len(isodepth_lines120))])
    
    isodepth_lines140 = isodepth_lines[isodepth_lines["Depth"] == -140]
    isodepth_lines140 = isodepth_lines140.set_index([np.arange(0, len(isodepth_lines140))])
    
    isodepth_lines160 = isodepth_lines[isodepth_lines["Depth"] == -160]
    isodepth_lines160 = isodepth_lines160.set_index([np.arange(0, len(isodepth_lines160))])
    
    isodepth_lines180 = isodepth_lines[isodepth_lines["Depth"] == -180]
    isodepth_lines180 = isodepth_lines180.set_index([np.arange(0, len(isodepth_lines180))])
    
    isodepth_lines200 = isodepth_lines[isodepth_lines["Depth"] == -200]
    isodepth_lines200 = isodepth_lines200.set_index([np.arange(0, len(isodepth_lines200))])
    
    # Plot
    for iso in [isodepth_lines20, isodepth_lines40, isodepth_lines60, isodepth_lines80, isodepth_lines100, isodepth_lines120, isodepth_lines140, isodepth_lines160, isodepth_lines180, isodepth_lines200]:
        map_lat_isodepth = []
        map_lon_isodepth = []
        for i in range(len(iso)):
            x, y = m(iso['Lon'][i], iso['Lat'][i])
            map_lon_isodepth.append(x)
            map_lat_isodepth.append(y)
        plt.plot(map_lon_isodepth, map_lat_isodepth, color='k', zorder = 4)

def plot_trench(m, lon_min, lon_max, lat_min, lat_max):
    # Trench line importation
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
    plt.plot(map_lon_trench, map_lat_trench, marker = 8, color='black', zorder = 2, markersize=9 )

    
def plot_bigEQ_japan(m):
    # Plot cities on the map
    lon_Tohoku = 142.861
    lat_Tohoku = 38.1035
    
    lon_foreshock = 143.279833
    lat_foreshock = 38.3285
    
    lon_deep2003 = 141.650667
    lat_deep2003 = 38.821
    

    depth_Tohoku = 23.74	
    depth_foreshock = 8.28 
    depth_deep2003 = 72.03	
    
    size_scale = 2000 #/2
    x, y = m(lon_Tohoku, lat_Tohoku)
    plt.scatter(x, y, marker='*', color='k',  s=size_scale, zorder=20, label='2011/03/11 Mw 9.0')
    #plt.text(x+dec, y+dec, '11/03/2011', color='k',fontsize=24, weight='bold', zorder=9)

    size_scale = 2000 #/2   
    x, y = m(lon_foreshock, lat_foreshock)
    plt.scatter(x, y, marker='*', color='grey',edgecolor='k',   s=size_scale, zorder=20, label ='2011/03/09 Mw 7.3')
    #plt.text(x+dec, y+dec, '09/03/2011', color='k',fontsize=24, weight='bold', zorder=9)
    
    size_scale = 2000 #/2
    x, y = m(lon_deep2003, lat_deep2003)
    plt.scatter(x, y, marker='*', color='blue', edgecolor='k', s=size_scale, zorder=20, label ='2003/05/26 Mw 7.0')
    #plt.text(x+dec, y+dec, '26/05/2003', color='k',fontsize=24, weight='bold', zorder=9)
    #plt.legend(fontsize=22)
    
def plot_hist(to_plot, x_label, y_label, my_color, ft):
    plt.hist(to_plot, color=my_color, edgecolor='black')
    plt.xlabel(x_label, fontsize=ft)
    plt.ylabel(y_label, fontsize=ft)
    plt.xticks(fontsize=ft, rotation=45)
    plt.yticks(fontsize=ft, rotation=45)
    
def plot_histc(to_plot, x_label, y_label, color, ft):
    plt.hist(to_plot, color=color, edgecolor='black')
    plt.xlabel(x_label, fontsize=ft)
    plt.ylabel(y_label, fontsize=ft)
    plt.xticks(fontsize=ft, rotation=45)
    plt.yticks(fontsize=ft, rotation=45)
    
# Polar histogram 
def plot_polar_hist(col, to_plot, title, bin_size, ax):
    a , b = np.histogram(to_plot, bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])   
    ax.bar(centers, a, Width=np.deg2rad(bin_size), bottom=0.0, color=col, edgecolor='k', align='edge')
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    #ax.set_ticks(fontsize=ft)
    ax.set_xticks(np.pi/180. * np.arange(0, 180, 30))
    ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize = 20)
    
    
# For ellipses
def confidence_ellipse(x, y, ax, n_std, facecolor, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor,zorder = 5, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Get list of lat and lon of all the events

def list_lin_lat(to_plot):
    lin_lat = []
    lin_lon = []
    lin_depth = []
    
    for i in range(len(to_plot)):
        for j in range(int(to_plot.nbre_events[i])):
            lin_lat.append(literal_eval(to_plot.Events_lat[i])[j])
            lin_lon.append(literal_eval(to_plot.Events_lon[i])[j])
            lin_depth.append(literal_eval(to_plot.Events_depth[i])[j])
    return(lin_lat, lin_lon, lin_depth)

# Get list of JST dates
from matplotlib import dates as mdates
def list_lin_time(to_plot):
    lin_time = [] 
    for i in range(len(to_plot)):
        for j in range(int(to_plot.nbre_events[i])):
            lin_time.append(literal_eval(to_plot.jst_timestamp[i])[j]) #Events_date_utc

    return(lin_time)

# Get list of depths
def list_lin_depth(to_plot):
    lin_depth = [] 
    for i in range(len(to_plot)):
        for j in range(int(to_plot.nbre_events[i])):
            lin_depth.append(literal_eval(to_plot.Events_depth[i])[j]) #Events_date_utc

    return(lin_depth)

def list_lin_whatever(dataframe, param): #param="..."
    lin_whatever = [] 
    for i in range(len(dataframe)):
        for j in range(int(dataframe.nbre_events[i])):
            lin_whatever.append(literal_eval(dataframe[param][i])[j]) #Events_date_utc

    return(lin_whatever)

############################### COSEISMIC TOHOKU #####################

def coseismic_tohoku():
    # Coseismic slip of Illapel : from Williamson 2017 https://doi.org/10.1002/2016JB013883
    coseismic = pd.read_csv('/Users/choulia/Documents/DOCTORAT/Article_correlations/Coseismic_Hooper2013.csv', names=["Lon", "Lat", "Depth(km)", "Dip", "Slip(m)", "Rake"], sep = ";", header = 1)
    
    # interpolation to have enough values for the plot
    from scipy.interpolate import griddata
    new_lon_1D = np.arange(min(coseismic["Lon"]), max(coseismic["Lon"]), 0.005 )
    new_lat_1D = np.arange(min(coseismic["Lat"]), max(coseismic["Lat"]), 0.005)
    
    new_lat = []
    new_lon = []
    for lat in new_lat_1D:
        for lon in new_lon_1D:
            new_lat.append(lat)
            new_lon.append(lon)
    
    interpolation          = griddata((coseismic["Lon"], coseismic["Lat"]), coseismic["Slip(m)"], (new_lon, new_lat), method='linear')
    coseismic_plot = pd.DataFrame(np.vstack((new_lon, new_lat, interpolation)).T, columns=['lon','lat','slip']).dropna()
    
    contour = coseismic_plot[((coseismic_plot["slip"] >= 14.7) & (coseismic_plot["slip"] <= 15.3))
                                     | ((coseismic_plot["slip"] >= 29.7) & (coseismic_plot["slip"] <= 30.3))
                                     | ((coseismic_plot["slip"] >= 44.7) & (coseismic_plot["slip"] <= 45.3))
                                     | ((coseismic_plot["slip"] >= 59.7) & (coseismic_plot["slip"] <= 60.3))]
    return(contour)

# Orthogonal projection to take events perpendicular to the trench 
def proj_ortho(lon_start_point,lat_start_point,slope_angle,all_lon_events,all_lat_events):
    from pyproj import Proj
    
    # geo --> cartesian
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



def interpolate_slab_kita(lon_min, lon_max, lat_min, lat_max):
    # Read Slab 2
    slab_japan_up = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/Japan_slab2_dep_02.24.18.xyz", names = ["Lon", "Lat", "Depth"])
    slab_japan_up = slab_japan_up.dropna()
    slab_japan_up = slab_japan_up.set_index([np.arange(0, len(slab_japan_up))])
    slab_japan_up = slab_japan_up[(slab_japan_up["Lon"] >= lon_min ) & (slab_japan_up["Lon"]<= lon_max) & (slab_japan_up["Lat"]>=lat_min) & (slab_japan_up["Lat"]<=lat_max)]
    slab_japan_up

    # Upper slab boundary : from Kita 2010
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


def list_lin_mag(to_plot):
    lin_mag = [] 
    for i in range(len(to_plot)):
        for j in range(int(to_plot.nbre_events[i])):
            lin_mag.append(literal_eval(to_plot.Events_mag[i])[j]) #Events_date_utc

    return(lin_mag)

def add_slab_depth_kita(catalog_japan):
    # Read Slab 2
    slab_japan_up = pd.read_csv("/Users/choulia/Documents/DOCTORAT/CODES/Autres/For_map/Slab/Japan_slab2_dep_02.24.18.xyz", names = ["Lon", "Lat", "Depth"])
    slab_japan_up = slab_japan_up.dropna()
    slab_japan_up = slab_japan_up.set_index([np.arange(0, len(slab_japan_up))])
    #slab_japan_up = slab_japan_up[(slab_japan_up["Lon"] >= lon_min ) & (slab_japan_up["Lon"]<= lon_max) & (slab_japan_up["Lat"]>=lat_min) & (slab_japan_up["Lat"]<=lat_max)]
    #slab_japan_up

    # Upper slab boundary : Kita 2010
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

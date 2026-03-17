import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from sklearn.cluster import DBSCAN
from numpy.linalg import eig
import os
from function_map_clusters import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import math

# Import events
#cat = pd.read_csv("/home/choulia/CATALOG_JMA_corrige.csv")
cat = pd.read_csv("/Users/choulia/Documents/Doctorat/catalogues/Japan/CATALOG_JMA_corrige.csv")

add_slab_depth(cat)
# Remove crustal events
cat = cat[cat["Distance to top slab"] <= 10]
cat = cat[cat.Depth >= 2]
cat.Date_UTC = pd.to_datetime(cat.Date_UTC)


# Period of study
#start_to_scan  = datetime(1995,1,1)
#end_to_scan    = datetime(2019,12,31)
start_to_scan  = datetime(2019,12,19)
end_to_scan    = datetime(2020,1,1)

duration       = 15  #duration of the window, hours
overlap        = 1/2 #fraction of the window that will overlap with the next one
number_window  = math.ceil((end_to_scan - start_to_scan)/timedelta(hours = duration*overlap)) #count the total number of windows
#condition_lmbd = 1600 # if max(eigenvalues) >= condition_lmbd, maybe cluster = lineament
#conditionR     = 3 # if R >= condition R, maybe cluster = lineament


# Map boundaries
lat_min        = 37
lat_max        = 40   
lon_min        = 139
lon_max        = 144
lat_coord2cart = 38

lon_Tohoku     = 142.861
lat_Tohoku     = 38.1035

lon_foreshock  = 143.279833
lat_foreshock  = 38.3285

lon_deep2003   = 141.650667
lat_deep2003   = 38.821


# Out directory
#out_dir = './With_migration/LAT40/OUT_nocontraction_lat40_cov/' + str(start_to_scan)[0:10] + '_' + str(end_to_scan)[0:10] + '_w' + str(duration) + '_step' + str(overlap) + 'xw'
out_dir = '/Users/choulia/Documents/DOCTORAT/CODES/JAPAN/Clustering//OUT/TEST_' + str(start_to_scan)[0:10] + '_' + str(end_to_scan)[0:10] + '_w' + str(duration) + '_step' + str(overlap) + 'xw'

isExist = os.path.exists(out_dir)
if not isExist:
    os.mkdir(out_dir)

# Empty arrays to fill during the loop (1 value per window)
stock_w_events    = np.zeros(number_window)
stock_w_mean_dist = np.zeros_like(stock_w_events)
stock_w_nclust    = np.zeros_like(stock_w_events)
stock_w_nlin      = np.zeros_like(stock_w_events)
# To fill with sequence
stock_w_lon       = []
stock_w_lat       = []
stock_w_depth     = []
    
# Empty lists to fill during the loop (1 value per cluster, and we don't know the number of clusters)
# Informations on the window associated to each cluster
list_start_window    = []
list_end_window      = []
list_nbre_events_w   = []
list_associated_w    = []
list_no_cluster_w    = []
# Basic informations on the cluster
list_start_cluster   = []
list_end_cluster     = []
list_n_events        = []
list_mean_dist_cluster = []
# Position and date of each event in the cluster
list_events_lon      = []
list_events_lat      = []
list_events_z        = []
list_x_cluster       = []
list_y_cluster       = []
list_date_cluster    = [] #UTC
list_jst_timestamp   = []
# Eigen vector, values, ratios etc
list_lmbd1           = []
list_lmbd2           = []
list_cov_xx          = []
list_cov_yy          = []
list_cov_xy          = []

list_cov_rot_xx      = []
list_cov_rot_yy      = []
list_cov_rot_xy      = []

list_length          = []
list_width           = []
list_eigen_vect_1    = []
list_eigen_vect_2    = []
list_scalar_prod     = []
list_aspectratio     = []
list_condR           = []
list_condlmbd        = []
list_cov_matrix      = []
# Lin = 1 if condR and conlmbd = 1
#list_check_lin       = []
# Orientation
angle_trench_arctan  = []
# Mags
list_mag_max         = []
list_mag_min         = []
list_mag_mean        = []
list_events_mag      = []
# Depths
list_depth_min       = []
list_depth_max       = []
list_depth_mean      = []
# For migration
list_x_PCA           = []
list_r2_regression   = []
list_a_regression    = []
list_b_regression    = []
# Distance from Tohoku, deep2003 
list_all_dist_Tohoku     = []
list_mean_dist_Tohoku    = []
list_all_dist_deep2003   = []
list_mean_dist_deep2003  = []
list_all_dist_foreshock  = []
list_mean_dist_foreshock = []
        

### LOOP ON EACH WINDOW

i     = 0 #count the number of windows
start = start_to_scan

while start < end_to_scan :
    print(i, '/', number_window)
    
    # Adjust times
    end = start + timedelta(hours = duration)
    list_start_window.append(start)
    list_end_window.append(end)

    # Select catalog, in space (map) and in time (window by window)
    selec = cat[(cat.Lon >= lon_min) & (cat.Lon <= lon_max) & (cat.Lat >= lat_min) & (cat.Lat <= lat_max) & (cat.Date_UTC >= start) & (cat.Date_UTC <= end)]
    selec = selec.reset_index().drop('index', axis=1)
    stock_w_events[i] = len(selec) #number of events on the window
    
    # Save positions of the events in the window (for futur plots ?)
    stock_w_lon.append(list(selec.Lon.values))
    stock_w_lat.append(list(selec.Lat.values))
    stock_w_depth.append(list(selec.Depth.values))
    
    # If more than 2 events on the window, search clusters 
    if len(selec) >= 3:

        # Import trench
        #trench   = pd.read_csv("all_segment", header=None, names =["Lon", "Lat"], delimiter=' ')
        trench   = pd.read_csv("/Users/choulia/Documents/Doctorat/Codes/Autres/For_map/all_segment", header=None, names =["Lon", "Lat"], delimiter=' ')
        trench   = trench[(lat_min - 1 <= trench.Lat) & (trench.Lat <= lat_max + 1) & (trench.Lon <= lon_max + 1) & (lon_min - 1 <= trench.Lon)]
        trench   = trench.sort_values(by=['Lon'])

        # Geographic coordinates --> cartesian 
        x  = selec.Lon * 111 * np.cos(np.radians(lat_coord2cart))
        y  = selec.Lat * 111
        xy = np.stack((x,y)).T
        
        x_trench = trench.Lon * 111 * np.cos(np.radians(lat_coord2cart))
        y_trench = trench.Lat * 111
        
        x_Tohoku = lon_Tohoku * 111 * np.cos(np.radians(lat_coord2cart))
        y_Tohoku = lat_Tohoku * 111
        
        x_foreshock = lon_foreshock * 111 * np.cos(np.radians(lat_coord2cart))
        y_foreshock = lat_foreshock * 111
        
        x_deep2003 = lon_deep2003 * 111 * np.cos(np.radians(lat_coord2cart))
        y_deep2003 = lat_deep2003 * 111

        ####### Rotation of 13° counterclockwise (to put the trench on the vertical, along y axis)        
        theta = np.radians(13);    # trench azimuth

        # Origin point for roation (set as the center of the trench)
        ox = min(x_trench) + (max(x_trench) - min(x_trench))/2
        oy = min(y_trench) + (max(y_trench) - min(y_trench))/2

        # Rotation matrix to calculate rotated x and y
        rx = np.zeros(xy.shape[0])
        ry = np.zeros(xy.shape[0])

        for jj in range(xy.shape[0]):
            rx[jj] = (ox + np.cos(theta) * (xy[jj][0] - ox) - np.sin(theta) * (xy[jj][1] - oy))
            ry[jj] = (oy + np.sin(theta) * (xy[jj][0] - ox) + np.cos(theta) * (xy[jj][1] - oy))
            rotated = np.column_stack((rx, ry))

        """ 
        ## /!\ IF NO ROTATION
        rx = x
        ry = y
        """

        ####### Contraction along rx axis (=along the axis perpendicular to the trench)
        rx = rx #/1.3

        ################################## DBSCAN #######################################
        # DBScan using rx and ry, to find clusters WITH ROTATION
        loc        = np.stack((rx, ry))
        clustering = DBSCAN(eps=40, min_samples=3).fit(loc.T)
        
        ####### PLOT
        #plt.figure(figsize=(10,10))
        #m = map_background(lon_min, lon_max, lat_min, lat_max)
        #plot_trench(m, lon_min, lon_max, lat_min, lat_max)
        #plot_isodepth_lines_japan(m)
        #plot_bigEQ_japan(m)
        #m_lon, m_lat = m(selec.Lon.values, selec.Lat.values)
        #plt.scatter(m_lon, m_lat, c='k', zorder = 2) # events before rotation in black (as in the catalogue)
        #plt.scatter(m_lon[clustering.labels_ > -1], m_lat[clustering.labels_ > -1], c="red", zorder = 3) #events in clusters
        #plt.show()
        ########### Caracterize clusters ON ORIGINAL DATA
        # Number of clusters
        stock_w_nclust[i] = max(clustering.labels_) + 1
        
        # LOOP ON ALL CLUSTERS
        for nn in np.arange(max(clustering.labels_) + 1):
            
            # Number of the 15h window in which is the cluster + number of events on that window
            list_associated_w.append(i)
            list_nbre_events_w.append(len(selec))

           
            # numero of the cluster
            list_no_cluster_w.append(nn)
            
            #position of the earthquakes inside the cluster
            x_cluster     = x[clustering.labels_ == nn].values
            y_cluster     = y[clustering.labels_ == nn].values

            rx_cluster     = pd.DataFrame(rx)[0][clustering.labels_ == nn].values
            ry_cluster     = pd.DataFrame(ry)[0][clustering.labels_ == nn].values

            lon_cluster   = selec.Lon.values[clustering.labels_ == nn]
            lat_cluster   = selec.Lat.values[clustering.labels_ == nn]
            depth_cluster = selec.Depth.values[clustering.labels_ == nn]
            list_x_cluster.append(x_cluster)
            list_y_cluster.append(y_cluster)
            list_events_lat.append(list(lat_cluster))
            list_events_lon.append(list(lon_cluster))
            list_events_z.append(list(depth_cluster)) 
            
            # number of events in the cluster
            list_n_events.append(len(x_cluster))
            
            # dates of earthquakes in the cluster
            list_date_cluster.append(list(selec.Date_UTC[clustering.labels_ == nn]))
            list_jst_timestamp.append(list(selec.timestamp_JST.values[clustering.labels_ == nn]))
           
            # magnitudes
            list_events_mag.append(list(selec.Mag.values[clustering.labels_ == nn]))

            # Average distance between all the events of the cluster
            all_dist_cluster = []
            for ev0 in range(len(x_cluster)):
                x0 = x_cluster[ev0]
                y0 = y_cluster[ev0]
                for ev1 in range(ev0 + 1, len(x_cluster)):
                    x1 = x_cluster[ev1]
                    y1 = y_cluster[ev1] 
                    all_dist_cluster.append(np.sqrt(((y0 - y1)**2 + (x0 - x1)**2)))
            list_mean_dist_cluster.append(np.mean(all_dist_cluster))
            
            # Distance of all the events from Tohoku
            all_dist_Tohoku = []
            for ev0 in range(len(x_cluster)):
                x0 = x_cluster[ev0]
                y0 = y_cluster[ev0]
                all_dist_Tohoku.append(np.sqrt(((y0 - y_Tohoku)**2 + (x0 - x_Tohoku)**2)))
            list_all_dist_Tohoku.append(list(all_dist_Tohoku))
            list_mean_dist_Tohoku.append(np.mean(all_dist_Tohoku))
            
            # Distance of all the events from 2003 intraslab earthquake
            all_dist_deep2003 = []
            for ev0 in range(len(x_cluster)):
                x0 = x_cluster[ev0]
                y0 = y_cluster[ev0]
                all_dist_deep2003.append(np.sqrt(((y0 - y_deep2003)**2 + (x0 - x_deep2003)**2)))
            list_all_dist_deep2003.append(list(all_dist_deep2003))
            list_mean_dist_deep2003.append(np.mean(all_dist_deep2003))
            
            # Distance of all the events from Tohoku foreshock
            all_dist_foreshock = []
            for ev0 in range(len(x_cluster)):
                x0 = x_cluster[ev0]
                y0 = y_cluster[ev0]
                all_dist_foreshock.append(np.sqrt(((y0 - y_foreshock)**2 + (x0 - x_foreshock)**2)))
            list_all_dist_foreshock.append(list(all_dist_foreshock))
            list_mean_dist_foreshock.append(np.mean(all_dist_foreshock))
            
            
            # Covariance matrix, eigen values and eigen vectors 
            cov_matrix           = np.cov(x_cluster,y_cluster)
            cov_matrix_rot       = np.cov(rx_cluster, ry_cluster)
            list_cov_matrix.append(list(cov_matrix))
            eigen_val,eigen_vect = eig(cov_matrix)
            # From the smallest to the largest
            idx                  = np.argsort(eigen_val)
            eigen_val_sorted     = eigen_val[idx] # eigenvalues ​​sorted in ascending order
            eigen_vect_sorted    = eigen_vect[:,idx] # associated vectors
            # Aspect ratio
            R                    = np.sqrt(eigen_val_sorted[1])/np.sqrt(eigen_val_sorted[0]) # R is the aspect ratio, i.e LENGTH / WIDTH
            list_length.append(np.sqrt(eigen_val_sorted[1])) # length, largest side
            list_width.append(np.sqrt(eigen_val_sorted[0])) # width, smallest side
            list_lmbd1.append(eigen_val[0]) #non sorted !!!
            list_cov_xx.append(cov_matrix[0,0])
            list_cov_yy.append(cov_matrix[1,1])
            list_cov_xy.append(cov_matrix[0,1])
            list_cov_rot_xx.append(cov_matrix_rot[0,0])
            list_cov_rot_yy.append(cov_matrix_rot[1,1])
            list_cov_rot_xy.append(cov_matrix_rot[0,1])
            list_lmbd2.append(eigen_val[1]) #non sorted !!! 
            list_eigen_vect_1.append(list(eigen_vect[:,0])) #non sorted !!!
            list_eigen_vect_2.append(list(eigen_vect[:,1])) #non sorted !!!
            
            # Scalar product    
            unit = np.array([np.sin(theta), np.cos(theta)]) # u = unitary vector pointing toward the trench
            scal = eigen_vect_sorted[0,1]*unit[0] + eigen_vect_sorted[1,1]*unit[1] #eigenvector [:,1], associated to the largest eingenvalue 
            list_scalar_prod.append(scal)
            
            # Converting from the dot product to the angle
            norm_unit          = np.sqrt(unit[0]**2 + unit[1]**2)
            norm_principal_dir = np.sqrt(eigen_vect_sorted[0,1]**2 + eigen_vect_sorted[1,1]**2)
            #angle_arccos       = np.rad2deg(np.arccos(scal/(norm_unit*norm_principal_dir)))
            angle_arctan       = np.rad2deg(np.arctan2(unit[1], unit[0]) - np.arctan2(eigen_vect_sorted[1,1], eigen_vect_sorted[0,1]))
            # better with arctan to keep sign info
            #angle_trench_arccos.append(angle_arccos)
            angle_trench_arctan.append(angle_arctan)
            
          
            # Duration of the clusters
            time_max = max(selec.Date_UTC.values[clustering.labels_ == nn])
            time_min = min(selec.Date_UTC.values[clustering.labels_ == nn])
            #duree    = (time_max - time_min)/ np.timedelta64(1, 'h')
            list_start_cluster.append(time_min)
            list_end_cluster.append(time_max)

            # Magnitudes 
            mag_max   = max(selec.Mag.values[clustering.labels_ == nn])
            mag_min   = min(selec.Mag.values[clustering.labels_ == nn])
            mag_mean  = np.mean(selec.Mag.values[clustering.labels_ == nn])
            delta_mag = mag_max - mag_min
            list_mag_min.append(mag_min)
            list_mag_max.append(mag_max)
            list_mag_mean.append(mag_mean)

            # Depths
            dep_max   = max(selec.Depth.values[clustering.labels_ == nn])
            dep_min   = min(selec.Depth.values[clustering.labels_ == nn])
            dep_mean  = np.mean(selec.Depth.values[clustering.labels_ == nn])
            delta_dep = dep_max - dep_min
            list_depth_min.append(dep_min)
            list_depth_max.append(dep_max)
            list_depth_mean.append(dep_mean)
            
            # Migration with PCA and Sklearn
            # Keep positions along the principal orientation with PCA
            data_for_pca        = np.stack((x_cluster, y_cluster)).T
            pca_1compo          = PCA(n_components=1).fit(data_for_pca) #reduction to 1 component
            transformed_data_1  = pca_1compo.transform(data_for_pca) #just 1D array
            inverse_transform_1 = pca_1compo.inverse_transform(transformed_data_1) #Back_rotated, to be oriented as original data
            x_to_keep           = inverse_transform_1[:,0] #Back_rotated, only x data
            list_x_PCA.append(list(x_to_keep))
            time                = selec.timestamp_JST.values[clustering.labels_ == nn] 
            # Fit between time and position, Linear regression with Sklearn
            # https://realpython.com/linear-regression-in-python/
            x_regress = time.reshape((-1, 1))
            y_regress = x_to_keep
            model     = LinearRegression().fit(x_regress, y_regress)
            r2        = model.score(x_regress, y_regress) #R2, coefficient of determination
            coeff_a   = model.coef_ #coefficient directeur = slope
            ord_b     = model.intercept_ #ordonnée à l'origine
            list_r2_regression.append(r2)
            list_a_regression.append(float(coeff_a))
            list_b_regression.append(ord_b)


   
    start = start + timedelta(hours = duration*overlap)
    i = i+1




#################### CREATE DATAFRAME WITH ALL THE INFORMATIONS #################

### INFORMATIONS ABOUT EACH WINDOW
df_windows          = pd.DataFrame((list_start_window, list_end_window, stock_w_events, stock_w_mean_dist, stock_w_nclust, stock_w_nlin)).T
df_windows.columns  = ['start', 'end', 'nbre_events', 'mean_dist', 'n_clust', 'n_lin']

df_windows["Lon"]   = np.zeros(len(df_windows))
df_windows["Lon"]   = df_windows["Lon"].astype('object')   #stock_w_lon
df_windows["Lat"]   = np.zeros(len(df_windows))
df_windows["Lat"]   = df_windows["Lat"].astype('object')   #stock_w_lat
df_windows["Depth"] = np.zeros(len(df_windows))
df_windows["Depth"] = df_windows["Depth"].astype('object') #stock_w_depth
for i in range(len(df_windows)):
    df_windows.at[i, "Lon"]      = stock_w_lon[i]
    df_windows.at[i, "Lat"]      = stock_w_lat[i]
    df_windows.at[i, "Depth"]    = stock_w_depth[i]
df_windows.to_csv(out_dir + '/' + "Windows.csv")




### INFORMATIONS ABOUT EACH CLUSTER
df_clusters                    = pd.DataFrame((list_associated_w, list_nbre_events_w, list_no_cluster_w, list_n_events, list_start_cluster, list_end_cluster, list_lmbd1, list_lmbd2, list_scalar_prod,list_aspectratio, list_mag_max, list_mag_min, list_mag_mean, list_depth_min, list_depth_max, list_depth_mean, list_condR, list_condlmbd, list_cov_xx, list_cov_yy, list_cov_xy, list_cov_rot_xx, list_cov_rot_yy, list_cov_rot_xy)).T
df_clusters.columns            =                ['window',    'nbre_events_window',    'no_cluster',      'nbre_events', 'start',           'end',            'lmbd1',      'lmbd2',    'scalar_prod',     'R',           'mag_max',     'mag_min',   'mag_mean',    'depth_min',     'depth_max',     'depth_mean',   'cond_R',     'cond_lmbd', 'cov_xx', 'cov_yy', 'cov_xy', 'cov_rot_xx', 'cov_rot_yy', 'cov_rot_xy']
df_clusters["delta_depth"]     = df_clusters.depth_max - df_clusters.depth_min
df_clusters["x_cluster"]       = list_x_cluster
df_clusters["y_cluster"]       = list_y_cluster
df_clusters["jst_timestamp"]   = list_jst_timestamp
df_clusters["R2_migration"]    = list_r2_regression
df_clusters["coeff_a_regress"] = list_a_regression
df_clusters["b_regress"]       = list_b_regression
df_clusters["Length"]          = list_length
df_clusters["Width"]           = list_width
df_clusters["Mean_dist_clust"] = list_mean_dist_cluster
df_clusters["Mean_dist_Toho"]  = list_mean_dist_Tohoku
df_clusters["Mean_dist_deep2003"]  = list_mean_dist_deep2003
df_clusters["Mean_dist_deep2003"]  = list_mean_dist_deep2003
df_clusters["Mean_dist_foreshock"] = list_mean_dist_foreshock
df_clusters["angle_trench_arctan"] = angle_trench_arctan

#To be able to read the lists after
df_clusters["Events_lat"]   = np.zeros(len(df_clusters))
df_clusters["Events_lat"]   = df_clusters["Events_lat"].astype('object')
df_clusters["Events_lon"]   = np.zeros(len(df_clusters))
df_clusters["Events_lon"]   = df_clusters["Events_lon"].astype('object')
df_clusters["Events_depth"] = np.zeros(len(df_clusters))
df_clusters["Events_depth"] = df_clusters["Events_depth"].astype('object')
df_clusters["Events_mag"]   = np.zeros(len(df_clusters))
df_clusters["Events_mag"]   = df_clusters["Events_mag"].astype('object')
df_clusters["eigen_vect_1"] = np.zeros(len(df_clusters))
df_clusters["eigen_vect_1"] = df_clusters["eigen_vect_1"].astype('object')
df_clusters["eigen_vect_2"] = np.zeros(len(df_clusters))
df_clusters["eigen_vect_2"] = df_clusters["eigen_vect_2"].astype('object')
df_clusters["x_PCA"]        = np.zeros(len(df_clusters))
df_clusters["x_PCA"]        = df_clusters["x_PCA"].astype('object')
df_clusters["duration"]            = (df_clusters["end"] - df_clusters["start"])/np.timedelta64(1, 'h')
df_clusters["Events_date_utc"]     = np.zeros(len(df_clusters))
df_clusters["Events_date_utc"]     = df_clusters["Events_date_utc"].astype('object')
df_clusters["Cov_matrix"]          = np.zeros(len(df_clusters))
df_clusters["Cov_matrix"]          = df_clusters["Cov_matrix"].astype('object')
df_clusters["All_dist_Toho"]       = np.zeros(len(df_clusters))
df_clusters["All_dist_Toho"]       = df_clusters["All_dist_Toho"].astype('object')
df_clusters["All_dist_deep2003"]   = np.zeros(len(df_clusters))
df_clusters["All_dist_deep2003"]   = df_clusters["All_dist_deep2003"].astype('object')
df_clusters["All_dist_foreshock"]   = np.zeros(len(df_clusters))
df_clusters["All_dist_foreshock"]   = df_clusters["All_dist_foreshock"].astype('object')

for i in range(len(df_clusters)):
    df_clusters.at[i, "Events_lat"]      = list_events_lat[i]
    df_clusters.at[i, "Events_lon"]      = list_events_lon[i]
    df_clusters.at[i, "Events_depth"]    = list_events_z[i]
    df_clusters.at[i, "Events_mag"]      = list_events_mag[i]
    df_clusters.at[i, "eigen_vect_1"]    = list_eigen_vect_1[i]
    df_clusters.at[i, "eigen_vect_2"]    = list_eigen_vect_2[i]
    df_clusters.at[i, "Events_date_utc"] = list_date_cluster[i]
    df_clusters.at[i, "Cov_matrix"]      = list_cov_matrix[i]
    df_clusters.at[i, "x_PCA"]           = list_x_PCA[i]
    df_clusters.at[i, "All_dist_Toho"]   = list_all_dist_Tohoku[i]
    df_clusters.at[i, "All_dist_deep2003"]  = list_all_dist_deep2003[i]
    df_clusters.at[i, "All_dist_foreshock"] = list_all_dist_foreshock[i]
df_clusters.to_csv(out_dir + '/' + "Clusters.csv")




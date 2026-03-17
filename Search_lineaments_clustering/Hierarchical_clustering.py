import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from functions import *
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from scipy.cluster.hierarchy import linkage, fcluster, to_tree

import sys
rep = '/Users/choulia/Documents/DOCTORAT/CODES/JAPAN/Clustering'
sys.path.append(rep)
from function_map_clusters import *


from ast import literal_eval

from sklearn.linear_model import LinearRegression



#################### HIERARCHICAL CLUSTERING ##########################


# Read DBSCAN results
directory                       = "/Users/choulia/Documents/DOCTORAT/CODES/JAPAN/Clustering/OUT/OUT_ISTOAR/With_migration/LAT40/OUT_nocontraction_lat40_cov/1995-01-01_2019-12-31_w15_step0.5xw_lmbd1600_R3/"
df_clusters_original            = pd.read_csv(directory + '/Clusters.csv', index_col=False, engine='python', encoding='utf-8',error_bad_lines=False) #info on each cluster
df_clusters_original["1/R"]     = 1/df_clusters_original["R"]
df_clusters_original.start      = pd.to_datetime(df_clusters_original.start)
date_start                      = datetime(2005,1,1) #to choose which period to study
date_end                        = datetime(2011,3,9)
df_clusters_original            = df_clusters_original[(df_clusters_original.start >= date_start) & (df_clusters_original.start <= date_end )]

# To have angles values between 0 and 180 (and not 32 and 212)
df_clusters_original.angle_trench_arctan.values[df_clusters_original.angle_trench_arctan.values >= 180] = df_clusters_original.angle_trench_arctan.values[df_clusters_original.angle_trench_arctan.values >= 180] - 180
df_clusters_original["sin_angle"] = np.sin(np.deg2rad(df_clusters_original.angle_trench_arctan.values))

# Import mantle wedge seismicity Lucile
with open('/Users/choulia/Documents/DOCTORAT/CATALOGUES/Japan/mantle_wedge_seismicity_catalog_Lucile_all_slabKita.txt', 'rb') as f:
    mantle_wedge = np.load(f)
    
cat_mantle_wedge = pd.DataFrame(mantle_wedge, columns=["Year", "Month", "Day", "Hour", "Minute", "Second", "Lat", "Lon", "Depth","Mag"])

# Create a column containing UTC date
date_UTC = []
for i in range(len(cat_mantle_wedge)):
    date_UTC.append(datetime(int(cat_mantle_wedge["Year"][i]), int(cat_mantle_wedge["Month"][i]), int(cat_mantle_wedge["Day"][i]), int(cat_mantle_wedge["Hour"][i]), int(cat_mantle_wedge["Minute"][i]), int(cat_mantle_wedge["Second"][i]))) 

date_UTC                     = np.asarray(date_UTC)
cat_mantle_wedge["Date_UTC"] = date_UTC


cat_mantle_wedge = cat_mantle_wedge[(cat_mantle_wedge.Date_UTC >= date_start) & (cat_mantle_wedge.Date_UTC <= date_end )  & (cat_mantle_wedge.Lon >= 139.5)] #& (cat_mantle_wedge.Mag >= 1.5 )

    
### Parameters to regroup clusters with hierarchical clustering

minEQ = 4 #7 #minimal number of earthquakse in the DSCAN clusters to study
df_clusters_minEQ   = df_clusters_original[df_clusters_original.nbre_events >= minEQ]
feature1 = "Length"
feature2 = "1/R"
feature3 = "sin_angle"  #"angle_trench_arctan"
feature4 = "delta_depth"

df_clusters_focus = df_clusters_minEQ[[feature1, feature2, feature3, feature4]] #.reset_index()#choose the criteria to regroup DBSCAN clusters
df_clusters_clean = df_clusters_focus #[(np.abs(stats.zscore(df_clusters_focus)) < 3).all(axis=1)]
df_clusters = df_clusters_minEQ[df_clusters_minEQ.index.isin(df_clusters_clean.index)]

# Re-scale the data and perform hierarchical clustering
dist = 5.3 
scaler              = MinMaxScaler()  #RobustScaler() #MinMaxScaler() #StandardScaler()
scaled_data         = scaler.fit_transform(df_clusters_clean) # Standardize features by removing the mean and scaling to unit variance ("données centrées réduites")
complete_clustering = linkage(scaled_data, method="ward", metric="euclidean") 
clusters            = fcluster(complete_clustering, dist, criterion='distance')

df_clusters["cluster"] = clusters
df_clusters[feature1 + "_scaled"], df_clusters[feature2 + "_scaled"], df_clusters[feature3 + "_scaled"], df_clusters[feature4 + "_scaled"] = [scaled_data[:,0], scaled_data[:,1], scaled_data[:,2], scaled_data[:,3]]

df_clusters = df_clusters.reset_index()


### Plot the associated dendogram
set_link_color_palette(palette_codes[0:100])
dendrogram(complete_clustering, color_threshold=dist,above_threshold_color='grey')
ax = plt.gca()
ax.tick_params(axis='y', which='major', labelsize=18)
plt.savefig("./2025_min_" + str(minEQ) + "EQ_sin/dendogram.png", transparent=True, bbox_inches='tight')
plt.show()


#################### STUDY OF EACH OBTAINED GROUP (Histograms, maps) ######################

########### Import the whole seismicity catalog to compare with lineaments

# TOHOKU
lat_min   = 37
lat_max   = 40.5   
lon_min   = 138.5
lon_max   = 144

### CATALOGUE REPEATERS UCHIDA
cat_repeat = pd.read_csv('/Users/choulia/Documents/Doctorat/Catalogues/Japan/Catalogue_repeaters_japan_Uchida2013.txt', delim_whitespace=True, skiprows = 1, names = ['Family','Time_JST','Lat','Lon','Depth', 'Mag', 'Largest_Coherence' , '2nd_largest_Coherence'])
cat_repeat = cat_repeat[(cat_repeat.Lon <= lon_max) & (cat_repeat.Lon>= lon_min) & (cat_repeat.Lat >= lat_min) & (cat_repeat.Lat <= lat_max)]
cat_repeat["Date_UTC"] = pd.to_datetime(cat_repeat.Time_JST , format='%Y%m%d%H%M%S') - np.timedelta64(9, 'h')
cat_repeat = cat_repeat[(cat_repeat.Date_UTC >= date_start) & (cat_repeat.Date_UTC <= date_end )]

fam = cat_repeat.drop_duplicates(subset=['Family'])

### CATALOGUE JMA
cat          = pd.read_csv("/Users/choulia/Documents/Doctorat/catalogues/Japan/CATALOG_JMA_corrige.csv")
cat.Date_UTC = pd.to_datetime(cat.Date_UTC)
cat_original = cat.copy(deep=True)
cat          = cat[(cat.Date_UTC >= date_start) & (cat.Date_UTC <= date_end) & (cat.Lon <= lon_max) & (cat.Lon >= lon_min) & (cat.Lat <= lat_max) & (cat.Lat >= lat_min)]
add_slab_depth_kita(cat)

cat          = cat[cat["Distance to top slab"] <= 10]
cat          = cat[cat["Depth"] >= 2]
cat.Date_UTC = pd.to_datetime(cat.Date_UTC)


# Catalogue large
lon_max_large = lon_max + 4
lon_min_large = lon_min - 4
lat_max_large = lat_max + 4
lat_min_large = lat_min - 4
cat_large = cat_original[(cat_original.Date_UTC >= date_start) & (cat_original.Date_UTC <= date_end) & (cat_original.Lon <= lon_max_large) & (cat_original.Lon >= lon_min_large) & (cat_original.Lat <= lat_max_large) & (cat_original.Lat >= lat_min_large)]

### Histograms parameters
days          = 60  # size of a bin, in days
limit_bins_precise = pd.DataFrame(pd.date_range(np.datetime64(date_start), date_end,  freq='2MS'), columns=['Date']) +  pd.DateOffset(days=8)
limit_bins_precise = pd.to_datetime(limit_bins_precise.Date).values

time_emphasis = limit_bins_precise[-3] #datetime(2011, 2, 1) #change the color of histogram bins from this date

### Seismicity evolution for the whole seismicity catalog
new_count_events_bins_precise = np.zeros(len(limit_bins_precise) - 1)

for i_date in range(len(limit_bins_precise) - 1):
    nbre_eq = len(cat[(limit_bins_precise[i_date] <= cat.Date_UTC) &  ( cat.Date_UTC < limit_bins_precise[i_date + 1]   )])
    new_count_events_bins_precise[i_date ] = nbre_eq

############# GROUP BY GROUP

stock_significativity = np.zeros(max(df_clusters.cluster.values))
stock_significativity_last_bin = np.zeros(max(df_clusters.cluster.values))
stock_slopes_real     = np.zeros(max(df_clusters.cluster.values))
stock_percent_high    = np.zeros(max(df_clusters.cluster.values))

stock_mean_feature1 = np.zeros(max(df_clusters.cluster.values))
stock_mean_feature2 = np.zeros(max(df_clusters.cluster.values))
stock_mean_feature3 = np.zeros(max(df_clusters.cluster.values))
stock_mean_feature4 = np.zeros(max(df_clusters.cluster.values))

stock_mean_feature1_scaled = np.zeros(max(df_clusters.cluster.values))
stock_mean_feature2_scaled = np.zeros(max(df_clusters.cluster.values))
stock_mean_feature3_scaled = np.zeros(max(df_clusters.cluster.values))
stock_mean_feature4_scaled = np.zeros(max(df_clusters.cluster.values))

stock_median_feature1 = np.zeros(max(df_clusters.cluster.values))
stock_median_feature2 = np.zeros(max(df_clusters.cluster.values))
stock_median_feature3 = np.zeros(max(df_clusters.cluster.values))
stock_median_feature4 = np.zeros(max(df_clusters.cluster.values))

### For each of the hierarchical clustering group, plots
for k in np.arange(1, max(df_clusters.cluster.values)+1): 
#for k in [6] : 

    ft = 32
    # map 
    fig = plt.figure(figsize=(12,20), dpi=300)
    gs  = GridSpec(nrows=2, ncols=2, width_ratios=[10, 1], height_ratios=[2, 1])
    ax  = fig.add_subplot(gs[0,0])
    m   = map_background(lon_min, lon_max, lat_min, lat_max)
    plot_trench(m, lon_min, lon_max, lat_min, lat_max)
    plot_bigEQ_japan(m)
    plt.title("GROUP "+ str(k), fontsize = ft)
    
    
    ### CLUSTER K
    ### Map

    cluster0 = df_clusters[df_clusters.cluster == k].reset_index()
        
    for i in range(len(cluster0)):  
       mx_ell, my_ell = m(np.array(literal_eval(cluster0.Events_lon[i])), np.array(literal_eval(cluster0.Events_lat[i])))
       confidence_ellipse(mx_ell, my_ell, ax, n_std=2 ,label=None, edgecolor='k', facecolor="white" , alpha = 0.2)
       
       mx_ell, my_ell = m(np.array(literal_eval(cluster0.Events_lon[i])), np.array(literal_eval(cluster0.Events_lat[i])))
       confidence_ellipse(mx_ell, my_ell, ax, n_std=2 ,label=None, edgecolor='k' , facecolor=palette[k-1] , alpha = 0.01)
    
    

    cluster_before = df_clusters[(df_clusters.cluster == k) & (df_clusters.start >= time_emphasis)].reset_index()
    for i in range(len(cluster_before)):  
       mx_ell, my_ell = m(np.array(literal_eval(cluster_before.Events_lon[i])), np.array(literal_eval(cluster_before.Events_lat[i])))
       confidence_ellipse(mx_ell, my_ell, ax, n_std=2 ,label=None, edgecolor='k' , facecolor=palette[k-1], alpha = 0.2)
    
    
    lin_lat_cluster0, lin_lon_cluster0, lin_depth_cluster0 = list_lin_lat(df_clusters[(df_clusters.cluster == k) & (df_clusters.start >= time_emphasis)].reset_index())

    # Associated seismicity
    mx_lin, my_lin = m(lin_lon_cluster0, lin_lat_cluster0)
    plt.scatter(mx_lin, my_lin, s=22, c='k', alpha=1, zorder = 7)
    
    
    # Mantle wedge seismicity
    # x, y = m(cat_mantle_wedge.Lon.values, cat_mantle_wedge.Lat.values)
    # plt.scatter(x, y, s=11, c='grey', alpha=0.7, zorder = 6, label='Mantle wedge seismicity')  
    
    # Repeaters from Uchida catalog
    # #x, y = m(fam.Lon.values, fam.Lat.values)
    # x, y = m(cat_repeat.Lon.values, cat_repeat.Lat.values)
    # plt.scatter(x, y, s=14, c='crimson', zorder = 7, label='repeaters') 
    
    # Coseismic slip Tohoku
    co_tohoku = coseismic_tohoku()
    size_scale = 2
    x, y = m(co_tohoku ["lon"].astype(float).values, co_tohoku ["lat"].astype(float).values)
    plt.scatter(x, y, c='k',s=size_scale, marker="o", zorder = 10)
    plt.plot(x[0:1], y[0:1], c='k', label='Coseismic slip Tohoku', zorder = 5, linewidth = 4)

    
    
    legend = plt.legend(fontsize = ft-8, loc = "upper left") #ft-8 #18
    legend.get_frame().set_alpha(0.3)
    legend.get_frame().set_facecolor("white")
    plt.title("FAMILY "+ str(k), fontsize = ft)
    fig.patch.set_alpha(0.0)

    ## Histogram 
    ft = 34
    
    df_clusters_criteria = df_clusters[df_clusters.cluster == k].reset_index()
    
    ax  = fig.add_subplot(gs[1,0])
    y_histo_precise, x_histo_precise, _ = ax.hist(df_clusters_criteria.start, bins=limit_bins_precise, color=palette[k-1], edgecolor = 'black', linewidth=2, label = "All lineaments", alpha = 0.1)
    ax.hist(df_clusters_criteria[(df_clusters_criteria.start >= time_emphasis)].start, bins=limit_bins_precise, color=palette[k-1],edgecolor='black', label = "from " + str(time_emphasis)[0:10])
    
    # Mean of the bins
    mean_bins = np.mean(y_histo_precise)
    std_bins  = np.std(y_histo_precise)
    
    ax.set_xlabel("Time (year)", fontsize=ft) 
    ax.set_ylabel("Number of clusters", fontsize=ft) 
    ax.tick_params(axis='x', which='major', labelsize=ft, rotation = 30, length=10)
    ax.tick_params(axis='y', which='major', labelsize=ft, length=10)
    # plot mean of the bins
    ax.plot(x_histo_precise[1:], np.ones(len(x_histo_precise[0:-1]))*mean_bins,'--', color='grey', linewidth = 5, alpha = 0.3)
    ax.text(x_histo_precise[-11], np.ones(len(x_histo_precise[0:-1]))[-10]*mean_bins + 0.1, "mean", fontsize=28, color="grey")
    # std
    ax.plot(x_histo_precise[1:], np.ones(len(x_histo_precise[0:-1]))*(mean_bins + 2*std_bins),color='grey', linewidth = 5, alpha = 0.3)
    ax.text(x_histo_precise[-11], np.ones(len(x_histo_precise[0:-1]))[-10]*(mean_bins + 2*std_bins) + 0.1,"mean + 2$\sigma$", fontsize = 28, color="grey")
    legend = ax.legend(fontsize = ft-4, loc="upper left")
    legend.get_frame().set_alpha(0.2)
    legend.get_frame().set_facecolor("white")
    ax2 = ax.twinx()
    ax2.plot(x_histo_precise[1:], new_count_events_bins_precise, c='grey')
    ax2.tick_params(axis='both', which='major', labelsize=ft, colors = "grey", rotation = 60, length=10)
    ax2.set_ylabel("Seismicity rate (n EQ / bin) ", fontsize=ft, c= "grey")
    ax2.scatter(mdates.date2num(cat_large[cat_large.Mag >= 6].Date_UTC.values), np.ones(len(cat_large[cat_large.Mag >= 6]))*max(new_count_events_bins_precise), c='red')

    
    fig.patch.set_alpha(0.0)
    #plt.title("Family " + str(k) , fontsize = ft-6) #+ ", normalized slope = " + str(round(coeff_a_real, 4))  
    fig.subplots_adjust(wspace=0, hspace=0)
    ##plt.savefig("./2025_min_" + str(minEQ) + "EQ_sin/Lineations_with_time_cluster" + str(k) + ".png", transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    stock_mean_feature1_scaled[k-1] = np.mean(df_clusters_criteria[feature1 + "_scaled"].values)
    stock_mean_feature2_scaled[k-1] = np.mean(df_clusters_criteria[feature2 + "_scaled"].values)
    stock_mean_feature3_scaled[k-1] = np.mean(df_clusters_criteria[feature3 + "_scaled"].values)
    stock_mean_feature4_scaled[k-1] = np.mean(df_clusters_criteria[feature4 + "_scaled"].values)


radar_chart = pd.DataFrame()
radar_chart["mean_" + feature1]   = stock_mean_feature1 / max(stock_mean_feature1)
radar_chart["mean_" + feature2]   = stock_mean_feature2 / max(stock_mean_feature2)
radar_chart["mean_" + feature3]   = stock_mean_feature3 / max(stock_mean_feature3)
radar_chart["mean_" + feature4]   = stock_mean_feature4 / max(stock_mean_feature4)


radar_chart_all = radar_chart.copy()
radar_chart_all["significativity"]   = stock_significativity / max(stock_significativity)


radar_chart_scaled = pd.DataFrame()
radar_chart_scaled["mean_" + feature1 + "_scaled"]   = stock_mean_feature1_scaled
radar_chart_scaled["mean_" + feature2 + "_scaled"]   = stock_mean_feature2_scaled
radar_chart_scaled["mean_" + feature3 + "_scaled"]   = stock_mean_feature3_scaled
radar_chart_scaled["mean_" + feature4 + "_scaled"]   = stock_mean_feature4_scaled 
radar_chart_scaled["significativity"]   = stock_significativity 



###### RADAR CHART

radar_chart_final_plot = radar_chart_scaled.drop("significativity", axis=1) 

fig = plt.figure(figsize=(12,12))
ax  = fig.add_subplot(111, projection="polar")

for k in np.arange(1, max(df_clusters.cluster.values)+1): #vgoes from 1 to 11
 
    theta = np.arange(len(radar_chart_final_plot.iloc[0]) + 1) / float(len(radar_chart_final_plot.iloc[0])) * 2 * np.pi
    values = radar_chart_final_plot.iloc[k-1].values
    values = np.append(values, values[0])
    l1, = ax.plot(theta, values, color=palette[k-1], marker="o")
    plt.yticks(size=26, color="grey")
    ax.tick_params(pad=50)
    # fill the area of the polygon
    ax.fill(theta, values, color=palette[k-1], alpha=0.2) #palette[k-1]
    plt.xticks(theta[:-1], ["Mean length", "Mean 1/R", "Mean angle", "Mean $\Delta$depth"], color='k', size=12)

##plt.savefig("./2025_min_" + str(minEQ) + "EQ_sin/Diagram_etoile.png", transparent=True, bbox_inches='tight')
plt.show()


### SAVE .CSV WITH LINEAMENTS CONTAINED IN A CHOSEN GROUP

date_today = datetime.today().strftime('%Y-%m-%d')
group2keep = df_clusters[df_clusters.cluster == 6]
# Remove added colums to have the same format as df_clusters_original to reuse it in another code
group2keep = group2keep.drop(['cluster', feature1 + '_scaled', feature2 + '_scaled', feature3 + '_scaled', feature4 + '_scaled', 'index'], axis=1)
group2keep.to_csv("/Users/choulia/Documents/DOCTORAT/CODES/JAPAN/Clustering/Catalogue_lineations_Japon_" + str(date_today) + ".csv")
print("mean delta depth (group significant) : ", np.mean(group2keep.delta_depth))
print("mean length (group significant) :",np.mean(group2keep.Length))
print("mean 1/R (group significant) :",np.mean(group2keep['1/R']))
print("mean angle to trench(group significant) :",np.mean(group2keep.angle_trench_arctan))


### Lineaments with migration pattern

migre_start_deep = group2keep[(group2keep.R2_migration >= 0.6) & (group2keep.coeff_a_regress > 0)]
migre_start_shal = group2keep[(group2keep.R2_migration >= 0.6) & (group2keep.coeff_a_regress < 0)]
migre_start_both = group2keep[(group2keep.R2_migration >= 0.6)]

print("Lineaments with migration  = ", len(migre_start_both), "/", len(group2keep), ", =", (len(migre_start_both)/len(group2keep))*100, "%")



##################################### MAP AND CROSS-SECTIONS LINEAMENTS ################################################



angle_value = 105 # angle of the cross-section with respect to the trench (0 = southward, 90 = westward)

# Projection Slab Kita on the cross section
slab_kita_interpolated = interpolate_slab_kita(lon_min, lon_max, lat_min, lat_max)
all_dist_slab_kita, all_x_proj_section_slab_kita = proj_ortho(lon_start_point=144,lat_start_point=38,slope_angle=angle_value,all_lon_events=slab_kita_interpolated.Lon.values,all_lat_events=slab_kita_interpolated.Lat.values)
slab_kita_interpolated["Dist_to_cross_km"] = all_dist_slab_kita /1000
slab_kita_interpolated["Dist_to_trench_km"] = all_x_proj_section_slab_kita /1000

############ LINEATION / LINEATION, MIGRATION

to_plot = group2keep[group2keep.R2_migration >= 0.6].reset_index() 
#to_plot = group2keep[group2keep.nbre_events >= 8] 
file = "Diffusivity_" + str(date_today)


for jj in [76]: #range(len(to_plot)):
    ind = jj
    lin_lat, lin_lon, lin_depth =             list_lin_lat(to_plot[ind:ind+1].reset_index()) #get lat, lon, depth of the events
    lin_time                    = np.asarray(list_lin_time(to_plot[ind:ind+1].reset_index()))
    lin_time_utc = pd.to_datetime(lin_time-719529, unit='D') - np.timedelta64(9, 'h')
    lin_x_PCA    = list_lin_whatever(to_plot[ind:ind+1].reset_index(), "x_PCA")
    size_scale = 300 
    

    ##### MAP ####
    fig, ax = plt.subplots(figsize=(10,20))
    m = map_background(lon_min, lon_max, lat_min, lat_max)
    plot_trench(m, lon_min, lon_max, lat_min, lat_max)
    plot_bigEQ_japan(m)
    
       
    mx_lin, my_lin = m(lin_lon, lin_lat)
    scatter    = plt.scatter(mx_lin, my_lin, c=mdates.date2num(lin_time_utc), cmap = 'hot', marker = 'o', edgecolor='k', s=size_scale, zorder =50)
    cbar       = plt.colorbar(scatter,fraction=0.035, pad=0.05)
    loc = mdates.AutoDateLocator()
    cbar.ax.yaxis.set_major_locator(loc)
    cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    cbar.ax.tick_params(labelsize=28)
    cbar.ax.yaxis.get_offset_text().set_fontsize(28) #for the small date on top   
    fig.show()
    
    
    size_scale = 300
    
    #### CROSS-SECTION #####
    # Project lineaments on the cross section
    all_dist_lin, all_x_proj_section_lin = proj_ortho(lon_start_point=144,lat_start_point=38,slope_angle=angle_value,all_lon_events=lin_lon,all_lat_events=lin_lat)
    dist_to_cross_km  = all_dist_lin /1000
    dist_to_trench_km = all_x_proj_section_lin /1000

    plt.figure(figsize=(8,4))
    scatter    = plt.scatter(dist_to_trench_km, - np.asarray(lin_depth), c=mdates.date2num(lin_time_utc), cmap = 'hot', marker = 'o', edgecolor='k', s=size_scale, zorder = 50)
    # Slab surface
    limite_slab = 1
    plt.plot(slab_kita_interpolated[slab_kita_interpolated.Dist_to_cross_km < limite_slab].Dist_to_trench_km.values, slab_kita_interpolated[slab_kita_interpolated.Dist_to_cross_km < limite_slab].Depth.values, c='k', linewidth = 1)
    # Limite Upper-Lower
    to_plot4    = slab_kita_interpolated[(slab_kita_interpolated.Dist_to_cross_km < limite_slab)]
    to_plot4.Depth = to_plot4.Depth - 23.5
    plt.plot(to_plot4.Dist_to_trench_km[to_plot4.Depth < -60], to_plot4.Depth[to_plot4.Depth < -60], '--', c= 'k', linewidth = 1)  
    cbar       = plt.colorbar(scatter,fraction=0.039, pad=0.05)#,cax=cax, label = cbar_name
    loc = mdates.AutoDateLocator()
    cbar.ax.yaxis.set_major_locator(loc)
    cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.get_offset_text().set_fontsize(20)
        
    plt.gca().invert_xaxis()
    plt.xticks(rotation=45, fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim((-150,0))
    plt.xlim((300,50))


    plt.xlabel("Distance to trench (km)", fontsize = 20)
    plt.ylabel("Depth (km)", fontsize = 20)
    #plt.title(str(np.round(to_plot[ind:ind+1].angle_trench_arctan.values[0])))
    ##plt.savefig(directory + "/Hierarchical_results_final/" + file + "/Lin_" + str(jj) + "_coupe.png", transparent=True, bbox_inches='tight', dpi = 300)
    plt.show()
    plt.close()
    
    ##### Plot DISTANCE TIME
    plt.figure(figsize=(7,5))
    scatter = plt.scatter(lin_time_utc, lin_x_PCA, c=mdates.date2num(lin_time_utc), cmap = 'hot', marker = 'o', s =50, edgecolor='k')
    cbar       = plt.colorbar(scatter,fraction=0.035, pad=0.05)#,cax=cax, label = cbar_name   
    loc = mdates.AutoDateLocator()
    cbar.ax.yaxis.set_major_locator(loc)
    cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    cbar.ax.tick_params(labelsize=28)
    cbar.ax.yaxis.get_offset_text().set_fontsize(28) #pour la ptite date en haut 
    plt.ylabel("Distance along main direction",fontsize=28)
    plt.xlabel("Time",fontsize=28)
    plt.xticks(rotation=45, fontsize = 20)
    plt.yticks(fontsize = 20)
    ##plt.savefig(directory + "/Hierarchical_results_final/" + file + "/Lin_" + str(jj) + "_distance_time.png", transparent=True, bbox_inches='tight', dpi = 300)
    plt.show()

    ##### Plot DISTANCE TIME WITH DIFFUSIVITY
    lin_x_PCA_m = np.asarray(lin_x_PCA) *(10**(3))
    dip_angle     = np.deg2rad(20)
    time_axis     = np.asarray((lin_time_utc - lin_time_utc[0]).total_seconds()) #seconds
    distance_axis = (lin_x_PCA_m - lin_x_PCA_m[0]) / np.cos(dip_angle) #meters

    plt.figure(figsize=(7,5))
    scatter = plt.scatter(time_axis, np.abs(distance_axis), c=mdates.date2num(lin_time_utc), edgecolor='k', cmap = 'hot',marker = 'o', s =50, zorder = 5)
    
    ### Find diffusivity value by trying to find the best fit of the diffusivity equation
    diffusivity = 600 
    plt.plot(time_axis,     np.sqrt(4*np.pi* diffusivity * time_axis), c='red', label="pore-pressure diffusion") 
    plt.scatter(time_axis,  np.sqrt(4*np.pi* diffusivity * time_axis), c='red', marker = 'o', s =50) 
       
    ### Linear regression
    x_regress = time_axis.reshape((-1, 1))
    y_regress = np.abs(distance_axis)
    model     = LinearRegression().fit(x_regress, y_regress)
    r2        = model.score(x_regress, y_regress) #R2, coefficient of determination
    coeff_a   = model.coef_ #coefficient directeur = slope
    ord_b     = model.intercept_ #ordonnée à l'origine
            
    plt.plot(x_regress, coeff_a * x_regress + ord_b, c='k', label="linear regression")
    
    plt.xlabel("Time (s)",fontsize=24)
    plt.ylabel("Distance (m)",fontsize=24)
    plt.xticks(rotation=45, fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.title("Lineation " + str(jj) + "\n Diffusivity =" +  str(diffusivity) + "m2/s", fontsize = 20) 
    ##plt.savefig(directory + "/Hierarchical_results_final/" + file + "/Lin_" + str(jj) + "_FIT_distance_time.png", transparent=True, bbox_inches='tight', dpi = 300)
    plt.show()
    
    
    print(" LINEAMENT n°", str(jj))
    print("R2 = ", r2)
    print("Diffusivity (m2/s) : ", diffusivity)
    print("Velocity    (m/s): ", np.abs(distance_axis[-1]) / time_axis[-1])
    print("Velocity   (km/h): ", np.abs(distance_axis[-1] * (10**(-3))) / (time_axis[-1] / 3600)) 
    print("Velocity  (m/day): ", np.abs(distance_axis[-1]) / (time_axis[-1] / (3600*24)))
    print("Duration (hours):", (time_axis[-1] / 3600))
    print("Duration (days):", time_axis[-1] / (3600*24))
    print("Distance   (m):", np.abs(distance_axis[-1]) )
    print("Distance   (km):", np.abs(distance_axis[-1] * (10**(-3))) )
    
	
### Mean mag lineaments
mag_lineaments = np.asarray(list_lin_mag(group2keep.reset_index()))
mag_lineaments = mag_lineaments[mag_lineaments>-1000]
print("Mean magnitude lineaments :", np.mean(mag_lineaments))




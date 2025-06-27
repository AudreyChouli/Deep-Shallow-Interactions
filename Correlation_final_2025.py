#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:03:42 2023

@author: choulia
"""

############################ IMPORTATION DES LIBRAIRIES #########################################
#################################################################################################


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

# Fonctions pour sélectionner sous-catalogues carte etc
from test_functions_2025 import * #_special_Bouchon2016

import multiprocessing
from multiprocessing import Pool
import os

#pd.set_option('max_columns', None)
#pd.set_option('max_rows', None)
#%matplotlib notebook


########################### SELECTION DES DEUX SOUS-CATALOGUES #####################################
#####################################################################################################


# Import the signal for the whole period (correlation + synthetic)
start_all   = datetime(2002,1,1) #datetime(1980,1,1)
end_all     = datetime(2011,3,9)

# Period on which the correlation is calculated
choose_start = datetime(2002,1,1) #datetime(1980,1,1) 
choose_end   = datetime(2011,3,9) 

t_min        = pd.to_datetime(choose_start)
t_max        = pd.to_datetime(choose_end)

# Period on which the "synthetics" = "not correlated signals" are picked
start_pick_synthetic = datetime(2002,1,1) #datetime(1980, 1, 1) 
end_pick_synthetic   = datetime(2011,3,9)


#list_sigma   = np.array([15*24, 7*24, 4*24])
#list_sigma   = np.arange(1, 31+1, 3)*24
list_sigma = np.arange(1,15)*24
mc_deep      = 1.5  #cutoff magnitude
mc_shallow   = 1.5


# Define the interest area
lat_min         = 37 #37.5
lat_max         = 40.5 #40  #39.5   
lon_min         = 138.5 #140
lon_min_shallow = lon_min # to change the area selected for shallow EQs
lon_max         = 144 #144


# Area for synthetics (okay if same area as windows are randomly picked ?)
lat_min_syn         = lat_min
lat_max_syn         = lat_max
lon_min_syn         = lon_min
lon_min_shallow_syn = lon_min_shallow
lon_max_syn         = lon_max


limit_depth_shallow = 40
limit_depth_deep    = 60
limit_end           = 400


#### Load catalog
catalog_japan             = pd.read_csv("/home/choulia/CATALOG_JMA_corrige.csv")
#catalog_japan["Date_UTC"] = catalog_japan["Date UTC"]
catalog_japan["Date_UTC"] = pd.to_datetime(catalog_japan["Date_UTC"])
catalog_japan             = add_slab_depth_kita(catalog_japan)
#catalog_japan             = catalog_japan[catalog_japan["Distance to top slab"]  <= 10 ]
catalog_japan             = catalog_japan[catalog_japan.Depth >= 2] # Remove crustal
catalog_japan             = catalog_japan[catalog_japan.Depth <= limit_end] # Max depth
# Zoom on the area to have less distances to cross section to compute
catalog_japan             = catalog_japan[(catalog_japan.Lat.values<lat_max) & (catalog_japan.Lat.values>lat_min) & (catalog_japan.Lon.values>lon_min)  & (catalog_japan.Lon.values<lon_max)]
# Seismicity band perpendicular to the trench
angle_value = 105
all_dist_orig, all_x_proj_section_orig  = proj_ortho(lon_start_point=144,lat_start_point=38,slope_angle=angle_value,all_lon_events=catalog_japan.Lon.values,all_lat_events=catalog_japan.Lat.values)
catalog_japan["Dist_to_cross_km"]  = all_dist_orig /1000
catalog_japan["Dist_to_trench_km"] = all_x_proj_section_orig /1000
catalog_japan = catalog_japan[catalog_japan.Dist_to_cross_km < 150]

tohoku                    = find_tohoku(catalog_japan)

#foreshock_tohoku          = catalog_japan[(datetime(2011,3,7) <= catalog_japan["Date_UTC"]) & (catalog_japan["Date_UTC"] < datetime(2011,3,10)) & (catalog_japan["Mag"] >= 7)]
#catalog_japan["Dist_foreshock_Tohoku"] = np.sqrt(((catalog_japan.Lat.values - foreshock_tohoku.Lat.values)*111)**2 + ((catalog_japan.Lon.values - foreshock_tohoku.Lon.values)*111*np.cos(np.radians(foreshock_tohoku.Lat.values)))**2)



# FOR SYNTHETICS (Correlation + Synthétiques), pour pouvoir piocher synthétiques dedans après
time_axis_all                      = pd.date_range(start = pd.to_datetime(start_all), end = pd.to_datetime(end_all), freq='h')

XX, EQ_shallow_syn, XX, XX, XX, XX, XX = selection_catalog(catalog_japan, t_min = start_all, t_max = end_all, mc = mc_shallow, limit_depth_deep = limit_depth_deep, limit_depth_shallow = limit_depth_shallow, lat_min = lat_min_syn, lat_max = lat_max_syn, lon_min = lon_min_syn, lon_min_shallow = lon_min_shallow_syn, lon_max = lon_max_syn)


# FOR CORRELATION 
time_axis       = pd.date_range(start=t_min, end=t_max, freq='h')

EQ_deep,    EQ_shallow_lost, EQ_between,       EQ_deep_large,    EQ_shallow_large_lost,  EQ_between_large, EQ_all_large      = selection_catalog(catalog_japan, t_min = t_min , t_max = t_max , mc = mc_deep,    limit_depth_deep = limit_depth_deep, limit_depth_shallow = limit_depth_shallow, lat_min = lat_min, lat_max = lat_max, lon_min = lon_min, lon_min_shallow = lon_min_shallow, lon_max = lon_max)

EQ_deep_lost, EQ_shallow,    EQ_between_lost, EQ_deep_large_lost, EQ_shallow_large,     EQ_between_large_lost, EQ_all_large_lost = selection_catalog(catalog_japan, t_min = t_min , t_max = t_max , mc = mc_shallow, limit_depth_deep = limit_depth_deep, limit_depth_shallow = limit_depth_shallow, lat_min = lat_min, lat_max = lat_max, lon_min = lon_min, lon_min_shallow = lon_min_shallow, lon_max = lon_max)

#EQ_shallow = EQ_shallow[EQ_shallow["Dist_foreshock_Tohoku"] <= 55]
#EQ_deep    = EQ_deep[EQ_deep["Dist_foreshock_Tohoku"] <= 222]



#display(foreshock_tohoku)


######################################## SHALLOW #######################################################
########################################################################################################

        
def f(sigma):
#for sigma in list_sigma:
    sigma  = int(sigma)        #hours
    window = int(sigma * 30)   #hours
    step   = int(1*sigma)      #hours sigma*2
    print("SIGMA = ", sigma, " h")
    print("WINDOW = ", window, " h, or", window/24, " d")

    out_dir = './2025_band_JMA_mcDeep' + str(mc_deep) + '_mcShallow' + str(mc_shallow) + '_limit' + str(limit_depth_shallow) + '-' + str(limit_depth_deep)+ '-' + str(limit_end) + '-from_' + str(t_min)[0:10] + '_to_' + str(t_max)[0:10] + '_SynStart' + str(start_pick_synthetic)[0:10] + '_SynStop' + str(end_pick_synthetic)[0:10] + '_Lon' + str(lon_min_shallow) + '-' + str(lon_min) + '-' + str(lon_max) + '_Lat'+ str(lat_min) + '-' + str(lat_max) + '_sigma' + str(sigma) + 'h_w' + str(window) + 'h_step'+ str(step) + 'h'
    isExist = os.path.exists(out_dir) 
    if not isExist:    
        os.mkdir(out_dir)
        
    ########### SHALLOW : convolution par une gaussienne ###############################

    time_shallow = EQ_shallow["Date_UTC"].values

    # Convolution
    y_axis_shallow = np.zeros(len(time_axis))
    for time_EQ in time_shallow:
        d              = (time_axis-time_EQ)/np.timedelta64(1, 'h')
        y_axis_shallow = y_axis_shallow +(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-d**2/(2*sigma**2))
    np.savetxt(out_dir + "/2025_Gaussian_smoothing_y_axis_shallow.txt", y_axis_shallow)
    pd.DataFrame(time_axis).to_csv(out_dir + "/2025_Gaussian_smoothing_time_axis.csv")

    """
    # Figure 
    plt.figure(figsize=(16,4))
    plt.plot(time_axis, y_axis_shallow, c='red')
    #plt.title("Lissage gaussien de la série des séismes superficiels")
    plt.xlabel('Time (years)', fontsize = 24)
    plt.ylabel('Amplitude', fontsize = 24)
    plt.xticks()
    plt.yticks(size=24)
    plt.legend()
    plt.show()
    """



    ############################################### DEEP ###############################################
    ####################################################################################################


    time_deep    = EQ_deep["Date_UTC"].values

    # Convolution par une gaussienne
    y_axis_deep = np.zeros(len(time_axis))
    for time_EQ in time_deep:
        d              = (time_axis-time_EQ)/np.timedelta64(1, 'h')
        y_axis_deep = y_axis_deep +(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-d**2/(2*sigma**2))
   
    np.savetxt(out_dir + "/2025_Gaussian_smoothing_y_axis_deep.txt", y_axis_deep)
    #y_axis_syn_deep_list = y_axis_syn_deep

    """
    # Figure 
    plt.figure(figsize=(16,4))
    plt.plot(time_axis, y_axis_deep, c='blue')
    #plt.title("Lissage gaussien de la série des séismes profonds ")
    plt.xlabel('Time (years)', fontsize = 24)
    plt.ylabel('Amplitude', fontsize = 24)
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.legend()
    plt.show()
    """



    ############################ CALCUL CORRELATION SERIES TEMPORELLES ################################
    ################################################################################################################


    ############################ SERIES DEEP ET SHALLOW REELLES ###################

    # Initial boundaries of the sliding window (from 2011 to 2000)
    start_window = time_axis[-1] - np.timedelta64(window, 'h')
    end_window   = start_window + np.timedelta64(window, 'h')

    # Lenght that have to be cover by the window (to create empty array of this size)
    len_temps    = ceil((((time_axis[-1] - time_axis[0])/np.timedelta64(1, 'h'))- window)/step)

    # To stock the boundaries of the windows with time (empty array doesn't work with dates)
    stock_start_window  = []
    stock_end_window    = []

    # To stock correlation values 
    stock_max_corr_window  = np.zeros((int(len_temps)))

    # To stock the number of events for each position of the window
    count_EQ_deep       = np.zeros_like(stock_max_corr_window) #nbr on the window
    count_EQ_shallow    = np.zeros_like(count_EQ_deep)

    count_EQ_deep_M6large       = np.zeros_like(stock_max_corr_window)
    count_EQ_shallow_M6large    = np.zeros_like(count_EQ_deep)
    count_EQ_between_M6large    = np.zeros_like(count_EQ_deep)
    count_EQ_all_M6large        = np.zeros_like(count_EQ_deep) 

    ##################### Loop on each position of the sliding window

    tt = 0 #to fill stock_start_window and stock_end_window. +1 each time the window moves

    print("START LOOP REAL DATA")
    #while end_window < time_axis[-1]:
    while start_window > time_axis[0]:

        print('--', start_window, '--', end_window, '/', time_axis[0])

        ################ SHALLOW AND DEEP :  FOCUS ON THE INTERESTING PART OF THE SERIE ###########################

        # Left boundary
        a = np.where((time_axis >= start_window) & (time_axis <= end_window))[0][0]
        
        # Right boundary
        b = np.where((time_axis >= start_window) & (time_axis <= end_window))[0][-1]
        
        # Do a copy to cut after
        y_axis_shallow_window_test = np.asarray(y_axis_shallow)
        y_axis_shallow_window      = np.copy(y_axis_shallow_window_test)
        y_axis_deep_window_test    = np.asarray(y_axis_deep)
        y_axis_deep_window         = np.copy(y_axis_deep_window_test)

        # Cut
        y_axis_shallow_cut = y_axis_shallow_window[a : b + 1]
        y_axis_deep_cut    =    y_axis_deep_window[a : b + 1]
        t_cut_test         =             time_axis[a : b + 1]

        """
        plt.figure(figsize=(8,8))
        plt.subplot(211)
        plt.plot(t_cut_test, y_axis_shallow_cut, c = 'red', label='shallow')
        plt.xticks(rotation=45)
        plt.subplot(212)
        plt.plot(t_cut_test, y_axis_deep_cut, c = 'blue', label='deep')
        plt.legend()
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.ylabel("Amplitude")
        plt.savefig(out_dir + '/FIG_start_' + str(start_window) + '_end' + str(end_window)+ '.png') #_mcDeep' + str(mc_deep) + '_mcShallow' + str(mc_shallow) + '_limit' + str(limit_depth_shallow) + '-' + str(limit_depth_deep)+ '-from_' + str(t_min)[0:10] + '_to_' + str(t_max)[0:10] + '_SynStart' + str(start_pick_synthetic)[0:10] + '_SynStop' + str(end_pick_synthetic)[0:10] + '_Lon' + str(lon_min_shallow) + '-' + str(lon_min) + '-' + str(lon_max) + '_Lat'+ str(lat_min) + '-' + str(lat_max) + '_sigma' + str(sigma) + 'h_w' + str(window) + 'h_step'+ str(step) + 'h.png')
        plt.show()
        plt.close()
        """
        ####################### CORRELATION OF DEEP - SHALLOW SERIES ON THE WINDOW ##########################@
        
        corr_window = np.corrcoef(y_axis_deep_cut, y_axis_shallow_cut)[0, 1]
        
        """
        # Figure
        plt.figure(figsize=(10,5))
        plt.plot(t_cut, corr_window, c='k')
        plt.xlabel('Time lag (h)')
        plt.ylabel('Correlation value')
        plt.title(str(start_window) + '--' + str(end_window))
        plt.show()
        """
        
        """
        # Figure
        plt.figure(figsize=(10,5))
        plt.plot(t_window, corr_window, c='k')
        #plt.plot(zoom_dt_window, np.zeros(len(zoom_dt_window)), '--', c='grey')
        plt.xlabel('Time lag (h)')
        plt.ylabel('Correlation value')
        plt.title(str(start_window) + '--' + str(end_window))
        plt.show()
        """   
        """  
        #print(len(tdf_std_deep_cut)/2)
        print(zoom_dt_window)
        # Figure
        plt.figure(figsize=(10,5))
        plt.plot(zoom_dt_window, zoom_corr_window, c='k')
        plt.plot(zoom_dt_window, np.zeros(len(zoom_dt_window)), '--', c='grey')
        plt.xlabel('Time lag (h)')
        plt.ylabel('Correlation value')
        plt.title(str(start_window) + '--' + str(end_window))
        plt.show()
        """ 
        stock_max_corr_window[tt] = corr_window

        stock_start_window.append(str(start_window))
        stock_end_window.append(str(end_window))

        """
        print("Max time lag = ", time_lag)
        print("------- Maximum correlation value = ", np.max(zoom_corr_window))
        """  
        count_EQ_shallow[tt] = count_events(catalog= EQ_shallow, start = start_window, end = end_window, m_min = None)
        count_EQ_deep[tt]    = count_events(catalog= EQ_deep  ,  start = start_window, end = end_window, m_min = None) 
        #count_EQ_between[tt] = count_events(catalog= EQ_between , start = start_window,end = end_window, m_min = None)

        count_EQ_shallow_M6large[tt] = count_events(catalog= EQ_shallow_large, start = start_window, end = end_window, m_min = 6) #les events au niveau du time_lag ajouté de part et d'autres sont = 0
        count_EQ_deep_M6large[tt]    = count_events(catalog= EQ_deep_large  , start = start_window,  end = end_window, m_min = 6) #les events au niveau du time_lag ajouté de part et d'autres sont = 0
        count_EQ_between_M6large[tt] = count_events(catalog= EQ_between_large  , start = start_window, end = end_window, m_min = 6) #les events au niveau du time_lag ajouté de part et d'autres sont = 0
        count_EQ_all_M6large[tt]    = count_events(catalog= EQ_all_large  , start = start_window, end = end_window, m_min = 6) 

        """
        if count_EQ_deep[tt] == 0 :
            print("WARNING 0 EQ deep")

            plt.figure()
            plt.plot(y_axis_deep_cut, c="blue")
            plt.show()

            plt.figure()
            plt.plot(y_axis_shallow_cut, c="red")
            plt.show()
        """    

        # Change sliding-window limits (go back in time)
        start_window = start_window - np.timedelta64(step, 'h')
        end_window   = end_window   - np.timedelta64(step, 'h')


        tt = tt+1

    # Create DataFrame    
    #stock_time_lag       = np.ones(len(stock_start_window)) * time_lag
    list_lon_min_shallow = lon_min_shallow*np.ones(len(stock_start_window))
    list_lon_min         = lon_min*np.ones(len(stock_start_window))
    list_lon_max         = lon_max*np.ones(len(stock_start_window))
    list_lat_min         = lat_min*np.ones(len(stock_start_window))
    list_lat_max         = lat_max*np.ones(len(stock_start_window))

    recap_real         = pd.DataFrame((list_lon_min_shallow,     list_lon_min ,      list_lon_max,       list_lat_min,    list_lat_max,       stock_start_window, stock_end_window, stock_max_corr_window, count_EQ_deep,   count_EQ_shallow,  count_EQ_deep_M6large ,  count_EQ_shallow_M6large,     count_EQ_between_M6large, count_EQ_all_M6large)).T
    recap_real.columns = [               'Lon min shallow(°)',   'Lon min(°)',       'Lon max(°)',        'Lat min(°)',     'Lat max(°)',      'Start',            'End',              'Real max corr',    'Count EQ deep',  'Count EQ shallow', 'Count EQ deep M6 large', 'Count EQ shallow M6 large',  'Count EQ between M6 large', 'Count EQ all M6 large']




    ###################################### CORRELATION AVEC SYNTHETIQUES ###########################################
    ################################################################################################################



    # Randomly picked windows to have uncorrelated signal

    random_synthetic   = 1000

    time_axis_syn      = pd.date_range(start = pd.to_datetime(start_pick_synthetic), end = pd.to_datetime(end_pick_synthetic), freq='h')

    y_axis_shallow_syn = np.zeros(len(time_axis_syn))
    time_shallow_syn   = EQ_shallow_syn["Date_UTC"].values

    # Pour l'instant rien de random
    for time_EQ in time_shallow_syn:
        d              = (time_axis_syn - time_EQ)/np.timedelta64(1, 'h')
        y_axis_shallow_syn = y_axis_shallow_syn +(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-d**2/(2*sigma**2))

    """
    # Figure 
    plt.figure(figsize=(16,4))
    plt.plot(time_axis_syn, y_axis_shallow_syn, c='orange', label="Shallow")
    plt.title("Lissage gaussien de la série des séismes superficiels - temps pour les synthétiques")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    """


    # Pour avoir les indexs de là où on peut piocher le début (ne pas piocher le début à la toute fin du signal )
    len_start_end      = len(time_axis_syn[(time_axis_syn  >= start_pick_synthetic) & (time_axis_syn <= end_pick_synthetic - timedelta(hours= window))])

    pick_amplitude_syn = y_axis_shallow_syn
    pick_time_syn      = time_axis_syn

    ##########@#@@ CORRELATION ################@@

    # Limites initiales de la fenêtre glissante
    start_window = time_axis[-1] - np.timedelta64(window, 'h')
    end_window   = start_window + np.timedelta64(window, 'h')

    # Longueur qu'il reste à parcourir à la fenêtre (pour pouvoir créer tableaux vides)
    len_temps = ceil((((time_axis[-1] - time_axis[0])/np.timedelta64(1, 'h'))- window)/step) 

    # Pour stocker les limites de la fenêtre glissante au cours de la boucle (array de zéros marche pas avec dates)
    stock_start_window_syn     = []
    stock_end_window_syn       = []

    # Tableaux vides pour stocker valeur de corrélation --> MOYENNE DES 1000
    stock_final_corr_max   = np.zeros((int(len_temps)))


    count_EQ_shallow_syn_all = []
    
    # Keep all the correlation values obtained to calculate probability
    pour_luck_apres = []
    
    tt = 0
    print("START LOOP SYNTHETICS")
    
    while start_window > time_axis[0]:
        print('--', start_window, '--', end_window, '/', time_axis[-1])

        stock_corr_time_lag_syn = np.zeros((random_synthetic))


        ################# Create synthetic
        # To stock the 1000 shallow "synthetics" picked in the signal

        y_axis_syn_shallow_list = []
        stock_start_syn         = []
        stock_end_syn           = []

        count_EQ_shallow_syn = np.zeros(random_synthetic)
        for ii in range(random_synthetic):
            start_syn_index = int(np.random.rand() * len_start_end)
            end_syn_index   = start_syn_index + window
            
            # On crée un synthétique
            synthetic_time_axis = pick_time_syn[start_syn_index : end_syn_index + 1]
            synthetic_amplitude = pick_amplitude_syn[start_syn_index : end_syn_index + 1]

            # On stocke le temps sur lequel il a été pioché
            stock_start_syn.append(synthetic_time_axis[0])
            stock_end_syn.append(synthetic_time_axis[-1])

            """
            plt.figure()
            plt.plot(synthetic_time_axis, synthetic_amplitude, color = "grey")
            plt.plot()
            """
            # On stocke son amplitude ???
            y_axis_syn_shallow_list.append(synthetic_amplitude)
            #On compte le nombre d'events sur ce synthétique
            count_EQ_shallow_syn[ii] = count_events(catalog = EQ_shallow_syn, start = synthetic_time_axis[0], end = synthetic_time_axis[-1], m_min = None)
        
        ############### on vient de créer et stocker 1000 synthétiques. On sort de la boucle
       
    
        #################### ON COMPARE AVEC DEEP EN PIOCHANT 1 SYN A CHAQUE FOIS  ############################
        # Limits
        a     = np.where((time_axis >= start_window) & (time_axis <= end_window))[0][0]
        b     = np.where((time_axis >= start_window) & (time_axis <= end_window))[0][-1]    
        
        y_axis_deep_window_test  = np.asarray(y_axis_deep)
        y_axis_deep_window       = np.copy(y_axis_deep_window_test)
        
        for k in range(random_synthetic):
            # 
            y_axis_syn_shallow        = y_axis_syn_shallow_list[k]
            #y_axis_syn_shallow_window = np.copy(y_axis_syn_shallow)
            
            y_axis_deep_cut = y_axis_deep_window[a: b + 1 ]
            #t_cut_test      = time_axis[a : b  + 1 ]
            
            corr_syn_window = np.corrcoef(y_axis_deep_cut, y_axis_syn_shallow)[0, 1]
            
            # Pour chaque synthétique
            stock_corr_time_lag_syn[k] = corr_syn_window
            
            """    
            if k <= 2 :

                if k == 0:
                    plt.figure(figsize=(8,4))
                    plt.plot(y_axis_deep_cut, c = 'blue')
                    plt.title("Deep")
                    plt.show()
                    
                plt.figure(figsize=(8,4))
                plt.plot(y_axis_syn_shallow, c = 'orange')
                plt.title("Shallow synthétique")
                plt.show()
            
            """
            
            
            """
             # If there is no event on a window
            if len(zoom_dt_syn_window[zoom_corr_syn_window==np.max(zoom_corr_syn_window)]) == 0 :
                stock_corr_time_lag_syn[k] = 0

                #print("No event on the window")
            else :  
            """  
            # Pour chaque synthétique
            stock_corr_time_lag_syn[k] = corr_syn_window


        count_EQ_shallow_syn_all.append(count_EQ_shallow_syn)

        # Pour chaque fenêtre ( stock_corr_time_lag_syn contient les corrélations max des 1000 synth sur 1 fenêtre)
        pour_luck_apres.append(stock_corr_time_lag_syn)

        stock_final_corr_max[tt] = np.mean(stock_corr_time_lag_syn) # mean maximum correlation value


        stock_start_window_syn.append(str(start_window))
        stock_end_window_syn.append(str(end_window))

        start_window = start_window - np.timedelta64(step, 'h')
        end_window   = end_window   - np.timedelta64(step, 'h')

        tt = tt+1



    ####################################### PROBABILITY ###########################################################
    ###############################################################################################################

    # DataFrame with synthetics
    recap_syn                          = pd.DataFrame((np.asarray(stock_start_window_syn).flatten(), np.asarray(stock_end_window_syn).flatten(),np.asarray(stock_final_corr_max).flatten() )).T
    recap_syn.columns                  =                         ['Start syn',                                  'End syn',                                'Syn mean max corr']
    recap_all                          = pd.concat((recap_real, recap_syn), axis=1)
    #recap_all


    # Calculate the probability to obtain with synthetics a better correlation than with real series

    tableau_all_corr = np.asarray(pour_luck_apres)
    proba_luck       = np.zeros(len(recap_all))

    for ii in range(len(recap_all)): #-1
        corr_real      = recap_all['Real max corr'].values[ii]
        #print(corr_real)
        corr_syn       = tableau_all_corr[ii, :]
        proba_luck[ii] = (len(corr_syn[corr_syn >= corr_real])/len(corr_syn)) * 100
        #print(corr_syn)

    recap_all['Proba luck'] = proba_luck
    recap_all["sigma(h)"]   = np.ones(len(recap_all)) * sigma
    recap_all["window(h)"]  = np.ones(len(recap_all)) * window
    recap_all["step(h)"]    = np.ones(len(recap_all)) * step
    recap_all["Mc_deep"]    = np.ones(len(recap_all)) * mc_deep
    recap_all["Mc_shallow"] = np.ones(len(recap_all)) * mc_shallow

    
    # !!! SAVE THE DATAFRAME !!!
    recap_all.to_csv(out_dir + '/Corrcoef_JMA_mcDeep' + str(mc_deep) + '_mcShallow' + str(mc_shallow) + '_limit' + str(limit_depth_shallow) + '-' + str(limit_depth_deep)+ '-from_' + str(t_min)[0:10] + '_to_' + str(t_max)[0:10] + '_SynStart' + str(start_pick_synthetic)[0:10] + '_SynStop' + str(end_pick_synthetic)[0:10] + '_Lon' + str(lon_min_shallow) + '-' + str(lon_min) + '-' + str(lon_max) + '_Lat'+ str(lat_min) + '-' + str(lat_max) + '_sigma' + str(sigma) + 'h_w' + str(window) + 'h_step'+ str(step) + 'h.csv', index = False)
    pour_luck_apres = np.asarray(pour_luck_apres)
    pd.DataFrame(pour_luck_apres).to_csv(out_dir + '/Corrcoef_JMA_LUCK' + str(mc_deep) + '_mcShallow' + str(mc_shallow) + '_limit' + str(limit_depth_shallow) + '-' + str(limit_depth_deep)+ '-from_' + str(t_min)[0:10] + '_to_' + str(t_max)[0:10] + '_SynStart' + str(start_pick_synthetic)[0:10] + '_SynStop' + str(end_pick_synthetic)[0:10] + '_Lon' + str(lon_min_shallow) + '-' + str(lon_min) + '-' + str(lon_max) + '_Lat'+ str(lat_min) + '-' + str(lat_max) + '_sigma' + str(sigma) + 'h_w' + str(window) + 'h_step'+ str(step) + 'h.csv', index = False)
    
    
    
from joblib import Parallel, delayed

Parallel(n_jobs=2, verbose=100)(delayed(f)(sigma) for sigma in list_sigma)
    

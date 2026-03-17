# Deep-Shallow-Synchronisation

Correlation_final_2025 : Calculate the significance of the correlation between shallow interplate earthquakes and intermediate-depth intraslab earthquakes through time, by comparing the cross-correlation values between the shallow and deep seismicity rates with random correlation values. 

Steps:
- Define the study period, the study area, and the parameters used to compute the seismicity rates
- Import the seismicity catalog
- Compute the shallow seismicity rate (convolution by a gaussian distribution)
- Compute the intermediate-depth seismicity rate (convolution by a gaussian distribution)
- Calculate the correlation values between the shallow and intermediate-seismicity rate on temporal sliding windows, stock them
- Pick 1000 random windows of shallow seismicity rates, and use them to calculate 1000 random correlation values with each window of the deep seismicity rate
- Calculate the probability to obtain with synthetics a better correlation than with real series

Return a dataframe with, for each sliding window, the values of the parameters used, the number of deep and shallow earthquakes on the window, and the significance of the correlation of the window. 

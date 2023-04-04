import numpy as np
import sys
import math
import os
import pandas as pd
from datetime import timedelta, datetime, date

################################################################################
# Converts hourly data to daily.
# Returns a disctionary containing "mean", "sum", and "max".
################################################################################
def hourly_to_daily(df_in):

  if (not (isinstance(df_in, pd.DataFrame))):

    index = df_in.index
    dt0 = index[0].date()
    dt1 = index[-1].date()

    index_daily = pd.date_range(dt0, dt1)
    
    arr_sum = np.zeros(len(index_daily))
    arr_mean = np.zeros(len(index_daily))
    arr_max = np.zeros(len(index_daily))

    for i in range(0, len(index_daily)):
      today = index_daily[i]
      dt_bgn = datetime(today.year, today.month, today.day, 0, 0)
      dt_end = datetime(today.year, today.month, today.day, 23, 59)

      tmp = df_in[dt_bgn:dt_end]

      arr_sum[i] = tmp.sum(skipna = True)
      arr_mean[i] = tmp.mean(skipna = True)
      arr_max[i] = tmp.max(skipna = True)

  else:
    raise Exception("Input must be dataframe")

  dict_out = {"sum": pd.DataFrame(data = arr_sum, index = index_daily).iloc[:,0],
              "mean": pd.DataFrame(data = arr_mean, index = index_daily).iloc[:,0],
              "max": pd.DataFrame(data = arr_max, index = index_daily).iloc[:,0]}

  return dict_out
  
#########################################################################
#   Compute metrics based on prediction and actual arrays.
#########################################################################
def compute_metrics(prediction, actual, ignore_nan = 0):
    
    assert(np.size(actual) == np.size(prediction))
    npt = np.size(actual)

    # Take care of NaN
    if (ignore_nan == 0): # there cannot be any NaN
        for i in range(0, npt):
            if(math.isnan(actual[i])):
                raise ValueError("NaN found in ACTUAL")
            if(math.isnan(prediction[i])):
                raise ValueError("NaN found in PREDICTION")
                
    elif (ignore_nan == 1): # NaN is removed from array.
        list_nan = []
        for i in range(0, npt):
            if(math.isnan(actual[i]) or math.isnan(prediction[i])):
                list_nan.append(i)

        actual = np.delete(actual, list_nan)
        prediction = np.delete(prediction, list_nan)
        
        assert(np.size(actual) == np.size(prediction))
        npt = np.size(actual)

    # Standard Deviation
    SD = np.sqrt( \
                (np.sum( (prediction - np.mean(prediction))**2 ) )  \
                /(npt-1)
                )

    # RMSD
    a = prediction - np.mean(prediction)
    b = actual - np.mean(actual)
    RMSD = np.sqrt( \
                  np.sum( (a-b)**2 )/npt
                  )

    #RSR
    a = np.sqrt(  \
                np.sum( (actual-prediction)**2 )
                )

    b = np.sqrt(  \
                np.sum( (actual-np.mean(actual))**2 )
                )
    RSR = a/b


    # mean
    MEAN = np.mean(prediction)

    # MAE
    MAE = np.sum(np.abs(prediction - actual))/npt


    #   R-squared
    #   https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/
    #
    sum_xy = np.sum(actual * prediction)
    sum_x2 = np.sum(prediction**2)
    sum_y2 = np.sum(actual**2)
    sum_x = np.sum(prediction)
    sum_y = np.sum(actual)

    # correlation coefficient
    R1 = (npt*sum_xy - sum_x * sum_y) \
        / np.sqrt( (npt*sum_x2 - sum_x**2) * (npt*sum_y2 - sum_y**2))

    R2 = R1 ** 2

    #
    # RMSE
    # https://statweb.stanford.edu/~susan/courses/s60/split/node60.html
    #
    sum0 = np.sum((prediction - actual)**2)
    RMSE = np.sqrt(sum0/npt)

    #
    # NSE
    # https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    #
    sum0 = np.sum((prediction - actual)**2)
    sum1 = np.sum((actual - np.mean(actual))**2)
    NSE = 1 - sum0/sum1

    #
    # Percent bias
    # https://rdrr.io/cran/hydroGOF/man/pbias.html#:~:text=Percent%20bias%20(PBIAS)%20measures%20the,values%20indicating%20accurate%20model%20simulation.
    #
    PBIAS = 100 * np.sum(prediction - actual)/np.sum(actual)


    dict_metric = {"SD": SD,
                  "RMSD": RMSD,
                  "RMSE": RMSE,
                  "RSR": RSR,
                  "R1": R1,
                  "R2": R2,
                  "NSE": NSE,
                  "PBIAS": PBIAS,
                  "MAE": MAE,
                  "MEAN": MEAN}
      
    return(dict_metric)

#########################################################################
#   Print metrics to csv file.
#########################################################################
def metrics_to_csv(cases: list, metrics:dict
                   , output_dir:str, output_file:str):

  list_metrics = list(metrics[cases[0]].keys())  # list of metrics
  path_out = os.path.join(output_dir, output_file)

  fout = open(path_out, "w")

  fout.write("Case,")

  prec = "%.3f" # precision

  # Column heading
  count = 1
  for metric in list_metrics:
    fout.write(metric)
    if (count < len(list_metrics)):
      fout.write(",")
    else:
      fout.write("\n")
    count = count + 1

  # Rows (metrics for each lcoation)
  for case in cases:
    fout.write(case + ",")

    count = 1
    for metric in list_metrics:
      fout.write(prec%metrics[case][metric])
      if (count < len(list_metrics)):
        fout.write(",")
      else:
        fout.write("\n")
      count = count + 1
  
  fout.close()
  
  return

###############################################################################
# Temperature conversion
###############################################################################
def convert_temp(temp_in, t_from: str, t_to:str):
  # From celsius
  if (t_from[0].upper() == 'C'):
    if (t_to[0].upper() == 'F'):
      temp_out = (temp_in * 9/5) + 32
    else: # change this to Kelvin later
      raise Exception("Check input")
  elif (t_from[0].upper() == 'F'):
    if (t_to[0].upper() == 'C'):
      temp_out = (temp_in - 32) / (9/5)
    else: # change this to Kelvin later
      raise Exception("Check input")
  else: # change this to Kelvin later
    raise Exception("Check input")
  return temp_out

###############################################################################
# Speed conversion
###############################################################################
def convert_speed(speed_in, speed_from: str, speed_to:str):

  # constants
  meters_in_mile = 1609.34
  seconds_in_hour = 3600

  # "clean up" the string
  speed_from = speed_from.replace("/","")

  # From m/sec
  if (speed_from[0:2].upper() == 'MS'):
    if (speed_to.upper() == 'MPH'):
      speed_out = speed_in / meters_in_mile * seconds_in_hour

    else:
      raise Exception("Check input")

  elif (speed_from[0:3].upper() == 'MPH'):
  
    if (speed_to.upper() == 'MS'):
      speed_out = speed_in * meters_in_mile / seconds_in_hour

    else:
      raise Exception("Check input")
    
  else:
    raise Exception("Check input")

  return speed_out

###############################################################################
# Radiation conversion
# https://www.nrcs.usda.gov/wps/portal/nrcs/detailfull/null/?cid=stelprdb1043619
###############################################################################
def convert_radiation(rad_in, rad_from: str, rad_to:str):

  # Constants
  c0 =  0.085985 # 1 Watt/m2 = 0.085985 Lang/hour
  hours_in_day = 24

  # "clean up" the string
  rad_from = rad_from.replace("/","")
  rad_to = rad_to.replace("/","")

  # From m/sec
  if (rad_from[0:5].upper() == 'WATTM'):
    if (rad_to[0:3].upper() == 'LYD'): # Langley/Day
      rad_out = rad_in * c0 * hours_in_day

    else:
      raise Exception("Check input")

  else:
    raise Exception("Check input")

  return rad_out
  
################################################################################
# Create auxiliary variables (e.g., tickmarks) for timeseries plots.
################################################################################
def pre_ts_plot(drange_dat, val, date_bgn_plot, date_end_plot
                   , month_interval: int):

  ##############################################################################
  # Bookkeeping
  ##############################################################################
  date_bgn_dat = datetime.strptime(drange_dat.iloc[0], "%Y-%m-%d").date()
  date_end_dat = datetime.strptime(drange_dat.iloc[-1], "%Y-%m-%d").date()

  drange_plot = pd.date_range(date_bgn_plot, date_end_plot, freq = "1D")
  
  ndays_pre = (date_bgn_plot - date_bgn_dat).days
  ndays_post = (date_end_plot - date_end_dat).days

  ind_xmin = ndays_pre
  ind_xmax = len(drange_dat) + ndays_post

  ##############################################################################
  #   label and indices for x-axis tick
  ##############################################################################
  ind = ind_xmin
  month_count = 0
  tickind = [ind]
  ticklabel = [date_bgn_plot.strftime("%b %Y")]
  for current_date in drange_plot:
      
      if (current_date > date_bgn_plot and current_date.day == 1):
          
          month_count = month_count + 1
      
          if(np.mod(month_count, month_interval) == 0):
              tickind.append(ind)
              ticklabel.append(current_date.strftime("%b %Y"))
              
      current_date = current_date + timedelta(days = 1)
          
      ind = ind + 1
      
  xset = np.arange(0, len(drange_dat))

  ##############################################################################
  #   Scan for any NaN
  ##############################################################################
  ind_nan = []
  for i in range(0, len(val)):
    if (math.isnan(val[i])):
      ind_nan.append(i)

  ind_nan = np.asarray(ind_nan)
  return [xset, [ind_xmin, ind_xmax], ticklabel, tickind, ind_nan]  
  
  
###############################################################################
# Function to compute wet-bulb temperature
###############################################################################
def twetbulb(rh, temp):
  # Reference: Equation (1) from
  # Roland Stull (2015), Wet-Bulb Temperature from Relative Humidity and Air Temperature
  # J. Appl. Meteor. Climatol. (2011) 50 (11): 2267â€“2269  
  # rh = relative humidity in FRACTION ranging between 0 and 1 (NOT %)
  # temp = temperature
  c1 = 0.151977
  c2 = 8.313659
  c3 = 1.676331
  c4 = 0.00391838
  c5 = 0.023101
  c6 = 4.686035

  rh100 = rh * 100
  twd = temp * np.arctan(c1 * np.sqrt(rh100 + c2)) \
      + np.arctan(temp + rh100) \
      - np.arctan(rh100 - c3) \
      + c4 * rh100**(3/2) * np.arctan(c5 * rh100) \
      - c6
  
  return twd  
  
###############################################################################
# Pressure conversion
###############################################################################
def convert_pressure(pressure_in, pressure_from: str, pressure_to:str):

  # constants
  Pa_in_mmHg = 133.32239
  Pa_in_inHg = 3386.38867
  Pa_in_kPa = 1000

  # From Pa
  if (pressure_from[0:2].upper() == 'PA'):
    if (pressure_to.upper() == 'MMHG'):
      pressure_out = pressure_in / Pa_in_mmHg
    elif (pressure_to.upper() == 'INHG'):
      pressure_out = pressure_in / Pa_in_inHg
    else:
      raise Exception("Check input")

  elif (pressure_from[0:3].upper() == 'KPA'):
    if (pressure_to.upper() == 'MMHG'):
      pressure_out = pressure_in * Pa_in_kPa / Pa_in_mmHg
    elif (pressure_to.upper() == 'INHG'):
      pressure_out = pressure_in * Pa_in_kPa / Pa_in_inHg
    else:
      raise Exception("Check input")

  else:
    raise Exception("Check input")

  return pressure_out  
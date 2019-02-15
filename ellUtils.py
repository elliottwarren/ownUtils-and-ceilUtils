# Ell's functions to make life easier
# Started Elliott Wed 15 Oct 2015


import datetime as dt
import numpy as np
from copy import deepcopy
from dateutil import tz
from bisect import bisect_left
import warnings

# Reading

def get_all_varaible_names(datapath):
    """
    get all variable names from netCDF file
    :param datapaths (list of str):
    :return: all_vars (list of str): all variable names
    """

    from os.path import isfile
    from netCDF4 import Dataset
    
    datafile = Dataset(datapath, 'r')
    
    all_vars = datafile.variables
    # convert from unicode to strings
    all_vars = [str(i) for i in all_vars]
    datafile.close()

    return all_vars

def csv_read(datapath):

    """
    Simple csv reader that puts data into lists of lists.
    :param datapath:
    :return: list of lists

    codecs helps prevent UTF-8 codes from emerging
    """

    from csv import reader
    import codecs

    # with open(datapath, 'r') as dest_f:
    with codecs.open(datapath, "r", encoding="utf-8-sig") as dest_f:
        data_iter = reader(dest_f,
                               delimiter=',',
                               quotechar='"')

        return [data for data in data_iter if data[0] != '']

def netCDF_info(datapath):

    """
    Display netcdf info including dimensions and such
    :param datapath:
    :return: printed list of variables, dimensions and units
    """

    from netCDF4 import Dataset#
    datafile = Dataset(datapath,'r')

    # headers
    print ''
    print '[name, units, shape]'

    for i in datafile.variables:

        meta = []

        # concatonate any relevent metadata


        # check if it has units before printing...
        if hasattr(datafile.variables[i], 'units'):
            meta += [datafile.variables[i].units]
        else:
            meta += ['']

        meta += [datafile.variables[i].shape]

        print [i] + meta
        # print [i,datafile.variables[i].units,datafile.variables[i].shape]

    datafile.close()

    return

#ToDo need to sort time out. Currently time gets processed even if it is not in vars. Need an if statement or two...
def netCDF_read(datapaths,vars='',timezone='UTC', returnRawTime=False, skip_missing_files=False):

    """
    Read in any variables for any netcdf. If vars are not given, read in all the data

    :param datapath (list of datapaths):
    :param vars: (list of strings)

    kwargs
    :param height: instrument height
    :return: raw (dictionary)
    """

    from netCDF4 import Dataset
    #from netCDF4 import MFDataset
    import numpy as np
    import datetime as dt
    from os.path import isfile
    import os

    # data prep
    # --------------

    # if just a single string given, put it into a single element list
    if type(vars) != list:
        try:
            vars = [vars]
        except TypeError:
            print 'Variables need to be a list of strings or '' to read all variables'
    
    if type(datapaths) != list:
        try:
            datapaths = [datapaths]
        except TypeError:
            print 'Datapaths need to be either a singular or list of strings'
    
    # find first existing datapath in [datapaths]
    first_file = []
    for i, d in enumerate(datapaths):
        if first_file == []:
            if isfile(d):
                first_file = d
                datapaths = datapaths[i:]
    
    # if no specific variables were given, find all the variable names within the first netCDF file
    if vars[0] == '':
        # ['latitude', 'longitude', 'time', 'model_level_number', 'level_height', 'level_sigma', 
        # 'air_temperature', 'specific_humidity', 'air_pressure', 'aerosol_for_visibility']
        vars = get_all_varaible_names(first_file)
    
    # Read
    # -----------
    
    # define array
    raw = {}

    for j, d in enumerate(datapaths):

        # debugging code
        #if j in range(0, 3000, 100):
        #   print d
        # print d

        # check file ecists. Raise error if all files need to be present (skip_missing_files == False)
        if (os.path.isfile(d) == False) and (skip_missing_files == False):
            raise ValueError(d + ' does not exist!')
        elif (os.path.isfile(d) == True):
            datafile = Dataset(d, 'r')

            # if first file and raw hasn't been populated
            if raw == {}:

                # Extract data and remove single dimension entries at the same time
                for i in vars:

#                     raw[i] = np.squeeze(datafile.variables[i][:])
                    raw[i] = datafile.variables[i][:]

                    # if masked array, convert to np.array
                    if isinstance(raw[i],np.ma.MaskedArray):
                        try:
                            mask = raw[i].mask # bool (True = bad/masked data)
                            raw[i] = np.asarray(raw[i], dtype='float32')
                            raw[i][mask] = np.nan
                        except:
                            warnings.warn('cannot convert '+i+'from masked to numpy!')


                # get time units for time conversion
                if 'time' in raw:
                    tstr = datafile.variables['time'].units
                    raw['protime'] = time_to_datetime(tstr, raw['time'])
                elif 'forecast_time' in raw:
                    tstr = datafile.variables['forecast_time'].units
                    raw['protime'] = time_to_datetime(tstr, raw['forecast_time'])

            # not the first file, append to existing array
            else:

                # Extract data and remove single dimension entries at the same time
                for i in vars:

#                     var_data = np.squeeze(datafile.variables[i][:])
                    var_data = datafile.variables[i][:]

                    # if masked array, convert to np.array
                    if isinstance(var_data, np.ma.MaskedArray):
                        try:
                            var_data = var_data.filled(np.nan)
                            mask = var_data.mask # bool (True = bad/masked data)
                            var_data = np.asarray(var_data, dtype='float32')
                            var_data[mask] = np.nan
                        except:
                            warnings.warn('cannot convert ' + i + 'from masked to numpy!')

                    raw[i] = np.append(raw[i], var_data)


                if 'time' in raw:
#                     todayRawTime = np.squeeze(datafile.variables['time'][:])
                    todayRawTime = datafile.variables['time'][:]
                    tstr = datafile.variables['time'].units
                    # append to stored 'protime' after converting just todays time
                    raw['protime'] = np.append(raw['protime'], time_to_datetime(tstr, todayRawTime))
                elif 'forecast_time' in raw:
#                     todayRawTime = np.squeeze(datafile.variables['forecast_time'][:])
                    todayRawTime = datafile.variables['forecast_time'][:]
                    tstr = datafile.variables['forecast_time'].units
                    raw['protime'] = np.append(raw['protime'], time_to_datetime(tstr, todayRawTime))

    # Sort Time names out
    # --------------------

    # allow 'rawtime' to be the unchanged time
    # make 'time' the processed time
    if 'time' in raw:
        raw['rawtime'] = deepcopy(raw['time'])
        raw['time'] = np.array(deepcopy(raw['protime']))
        #raw['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in raw['time']])
        del raw['protime']
    elif 'forecast_time' in raw:
        raw['rawtime'] = deepcopy(raw['forecast_time'])
        raw['time'] = np.array(deepcopy(raw['protime']))
        del raw['protime']

    if returnRawTime == False:
        if 'rawtime' in raw:
            del raw['rawtime']

    # close file
    datafile.close()

    return raw

# processing

def dec_round(a, decimals=1):

    """
    Round to nearest decimal place
    :param a:
    :param decimals:
    :return: rounded number
    """

    from numpy import around

    return around(a-10**(-(decimals+5)), decimals=decimals)

def to_percent(y, position):

    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.

    from matplotlib import rcParams

    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'     
    
#==============================================================================

def nearestDate(dates, pivot):

  date = min(dates, key=lambda x: abs(x - pivot))

  # no easy way for numpy, given it doesn't play nice with datetimes.
  # nor easy to check if numpy array, hence check if list and if it isn't, convert it.
  if isinstance(dates, list):
      idx = dates.index(date)
  else:
      dates = dates.tolist()
      idx = dates.index(date)

  # difference in dates
  diff = date - pivot

  return date, idx, diff

def nearest(array, pivot):

    """
    # find the nearest value, idx and difference to a list. Works with list or numpy arrays
    :param array:
    :param pivot:
    :return: value, idx, diff
    """

    value = min(array, key=lambda x: abs(x - pivot))

    # no easy way for numpy, given it doesn't play nice with datetimes.
    # nor easy to check if numpy array, hence check if list and if it isn't, convert it.
    if isinstance(array, list):
      idx = array.index(value)
    else:
      array = array.tolist()
      idx = array.index(value)

    # difference
    diff = value - pivot

    return value, idx, diff

def nannearest(array, pivot):

    """
    # find the nearest value, idx and difference to a list. Works with list or numpy arrays
    :param array:
    :param pivot:
    :return: value, idx, diff
    """

    if all(np.isnan(array)) | np.isnan(pivot):
        return np.nan, np.nan, np.nan
    else:

        value = min(array, key=lambda x: abs(x - pivot))

        # no easy way for numpy, given it doesn't play nice with datetimes.
        # nor easy to check if numpy array, hence check if list and if it isn't, convert it.
        if isinstance(array, list):
          idx = array.index(value)
        else:
          array = array.tolist()
          idx = array.index(value)

        # difference
        diff = value - pivot

        return value, idx, diff

def merge_dicts(dict1, dict2):

    """
    Fairly simple attempt to merge the lowest dicts togethers. Only works with 2 tier system, with these specific
    names. Ideally wants a more elegent/general solution.

    :param dict1:
    :param dict2: appeneded ONTO dict 1
    :return final_dict: merged dictionary
    """

    final_dict = {}

    for site, site_data in dict1.iteritems():

        # set up site within final dict
        final_dict[site] = {}

        for var, var_data in site_data.iteritems():

            if var == 'time':
                final_dict[site][var] = var_data + dict2[site][var]
            elif (var == 'level_height') | (var == 'height'):
                final_dict[site][var] = var_data
            else:
                final_dict[site][var] = np.ma.vstack((var_data, dict2[site][var]))

    return final_dict

def moving_average(interval, window_size):

    """
    Moving average

    interval is the data[:,1]
    http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """

    from numpy import convolve
    import numpy

    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def replace_all(str_list, value):

    """ replace all defined strings within a list"""

    for i in str_list:
        text = text.replace(i, value)
    return text

def linear_interpolation(y):

    """Linearly interpolate the data over nans"""

    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y


def binary_search_orig(array, element, lo=0, hi=None):  # can't use a to specify default for hi

    # Search through a list  and get idx value out

    hi = hi if hi is not None else len(array)  # hi defaults to len(a)
    pos = bisect_left(array, element, lo, hi)  # find insertion position
    return (pos if pos != hi and array[pos] == element else -1)  # don't walk off the end

def binary_search(array, element, lo=0, hi=None):

    # FAST APPROXIMATE Search through a list and get idx value out

    hi = hi if hi is not None else len(array)  # hi defaults to len(a)
    pos = bisect_left(array, element, lo, hi)  # find insertion position
    return (pos if pos != hi else -1) # don't walk off the end

def binary_search_nearest(array, element, lo=0, hi=None):  # can't use a to specify default for hi

    """
    PRECISE search on a sorted array - far faster than eu.nearest (speed increase is greatest on large arrays).

    bisect_left will return the index where to insert item x in list a, assuming a is sorted. Idx will either be correct
    or 1 too high. Therefore use abs(array[idx] - element) < abs(array[idx-1] - element) to check and make sure the
    idx with the nearest value to 'element' is used
    """

    hi = hi if hi is not None else len(array)  # hi defaults to len(a)
    pos = bisect_left(array, element, lo, hi)
    idx = (pos if pos != hi else -1) # find insertion position

    if abs(array[idx] - element) < abs(array[idx-1] - element):
        return idx
    else:
        return idx-1

# statistics

def croscorr(data1, data2, normalise=False):
    """
    #==============================================================================
    # 1. Normalised cross correlation with NaNs present
    #==============================================================================
    # 15 Oct 2015
    # Does a full, independent cross correlation (correlation between two different
    # time series).
    # Designed to deal with data containing NaNs but keeping lag position (tau)
    # normalises input data in the function.
    # [a] is fixed, and [b] is lagged across it.


    # inputs:
    # a = timeseries 1; b = timeseries 2
    # var_mat = list of variables names as they appear in the netCDF file
    #
    # outputs:
    # obs = list containing np.arrays of each variable, as defined by var_mat.

    # a and b need to be equal length

    # sepererate way up and way down for human readability.
    # could combine t and d loops...

    :param data1:
    :param data2:
    :param normalise:
    :return:
    """

    import numpy as np

    ## 1.1 normalise data for cross-correlation
    # -----------------------------------------------

    if normalise == True:
        # Initialise then fill
        # technique taken from: http://stackoverflow.com/questions/5639280/
        #   why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize

        # normalise data is diffferent units or magnitude responses are different.

        # BSCnorm = (BSC - mean(BSC)) / std(BSC)
        data1 = ((data1 - np.nanmean(data1))
                 / (np.nanstd(data1 * len(data1))))

        # no need to multiple by length here
        data2 = ((data2 - np.nanmean(data2))
                 / (np.nanstd(data2)))

    # force data2 offset by 10 mins
    # bsc_avg_h2 = np.hstack([bsc_avg_h1[len(bsc_avg_h1)-10:],bsc_avg_h1[:len(bsc_avg_h1)-10]])
    # data2 = np.hstack([data2[len(data2)-0:],data2[:len(data2)-0]])

    ## 1.2 Carry out cross-correlation
    # -----------------------------------------------

    # keep them seperate so it is simpler to read

    # define output correlation array
    crscor = np.empty((len(data1) * 2) - 1)
    crscor[:] = np.nan

    ## 1.3 Way up
    # ----------------

    # good for tau_-N+1 to tau_-1
    for t in range(len(data1) - 1):
        #    cor = np.correlate(a[:t+1],b[len(a)-t-1:])

        # extract out a and b
        a = data1[:t + 1]
        b = data2[len(data2) - t - 1:]

        # boolean vector giving True if neither pair of a or b are nan
        # got to be a better way to store and move this, rather than recalculate.
        vec = np.array([True if ~np.isnan(a[i]) and ~np.isnan(b[i])
                        else False
                        for i in range(len(a))])

        # correlate a and b, not including nans
        crscor[t] = np.correlate(a[vec], b[vec])

    # 1.4 Mid point
    # ----------------
    # tau_0

    vec = np.array([True if ~np.isnan(data1[i]) and ~np.isnan(data2[i])
                    else False
                    for i in range(len(data1))])

    # dot product
    crscor[len(data1) - 1] = np.correlate(data1[vec], data2[vec])

    # 1.5 Way down
    # ----------------
    # good for tau_1 to tau_N-1
    for d in range(len(data1) - 1):
        # extract out a and b
        a = data1[d + 1:]
        b = data2[:len(data2) - d - 1]

        vec = np.array([True if ~np.isnan(a[i]) and ~np.isnan(b[i])
                        else False
                        for i in range(len(a))])

        # dot product
        crscor[len(data1) + d] = np.correlate(a[vec], b[vec])

    # return cross correlation vector
    return crscor

def moving_average_old(a, n):
    from numpy import cumsum

    ret = cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pstar(p):

    """
    Turn p into stars based on a threshold for significance
    :param p:
    :return: p_star (**, * or '')
    """

    if p <= 0.01:
        p_star = '**'
    elif p <= 0.05:
        p_star = '*'
    else:
        p_star = ''

    return p_star

# distributions

def normpdf(x, mean, sd):

    """
    Gives the probability of y, given a value of x, assuming a gaussian distribution
    :param x:
    :param mean:
    :param sd:
    :return:

    unit of p(y) is a fraction
    """

    from math import exp

    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = exp(-(float(x) - float(mean)) ** 2 / (2 * var))

    return num / denom

def inv_normpdf(x, mean, sd):

    """
    Gives the probability of y, given a value of x, assuming a gaussian distribution
    :param x:
    :param mean:
    :param sd:
    :return:

    unit of p(y) is a fraction
    """

    from math import exp

    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = exp(-(float(x) - float(mean)) ** 2 / (2 * var))

    

    return num / denom

# datetime processing

def dateList_to_datetime(dayList):

    """ Convert list of string dates into datetimes - very rigid methodology """
    from numpy import array

    datetimeDays = array([dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8])) for d in dayList])

    return datetimeDays

def date_range(start_date, end_date, increment, period):

    """
    # start_date is a datetime variable
    # end_date is a datetime variable
    # increment is the time e.g. 10, 20, 100
    # period is the time string e.g. 'seconds','minutes', 'days'

    :param start_date:
    :param end_date:
    :param increment:
    :param period:
    :return:

    period is the plural of the time period e.g. days and not day, minuts and not minute. If the singular is given,
    the code will replace it
    """

    # replace period string with plural, if it was singlar
    # function cannot do hours so convert it to minutes
    if period == 'day':
        period = 'days'
    elif period == 'minute':
        period = 'minutes'
    elif period == 'second':
        period = 'seconds'
    elif period == 'hour':
        period = 'minutes'
        increment *= 60.0

    from dateutil.relativedelta import relativedelta

    result = []
    nxt = start_date
    delta = relativedelta(**{period:increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return np.array(result)

def time_to_datetime(tstr, timeRaw):

    """
    Convert 'time since:... and an array/list of times into a list of datetimes

    :param tstr: string along the lines of 'secs/mins/hours since ........'
    :param timeRaw:
    :return: list of processed times
    """

    import datetime as dt
    
    # sort out times and turn into datetimes
    # tstr = datafile.variables['time'].units
    tstr = tstr.replace('-', ' ')
    tstr = tstr.split(' ')
    
    # Datetime
    # ---------------
    # create list of datetimes for export
    # start date in netCDF file
    start = dt.datetime(int(tstr[2]), int(tstr[3]), int(tstr[4]))

    # get delta times from the start date
    # time: time in minutes since the start time (taken from netCDF file)
    if tstr[0] == 'seconds':
        delta = [dt.timedelta(seconds = int(timeRaw[i])) for i in np.arange(0, timeRaw.size)] # len(timeRaw)

    elif tstr[0] == 'minutes':
        delta = [dt.timedelta(seconds=timeRaw[i]*60) for i in np.arange(0, timeRaw.size)]

    elif tstr[0] == 'hours':
        # delta = [dt.timedelta(seconds=timeRaw[i]*3600) for i in np.arange(0, timeRaw.size)]
        delta = [dt.timedelta(seconds=t*3600) for t in timeRaw]

    elif tstr[0] == 'days':
        delta = [dt.timedelta(days=timeRaw[i]) for i in np.arange(0, timeRaw.size)]


    if 'delta' in locals():
        return [start + delta[i] for i in np.arange(0, timeRaw.size)]
    else:
        print 'Raw time not in seconds, minutes, hours or days. No processed time created.'
        return

def time_match_datasets(xdata, ydata):

    """
    Time match xdata and ydata
    :param xdata:
    :param ydata
    :return:
    """

    def time_match_process_single_site(data, time_range):

        """
        Time match up a single set of observations to the time_range
        IF at the same time resolution
        :param data:
        :param time_range:
        :return:
        """

        # identify keys are with and without a time dimension (but not the time variabile itself).
        #   With: keep the key
        #   Without: just make a copy of its data into data_pro
        time_variable_keys = []
        none_time_variable_keys = []
        for key, item in data.iteritems():
            # if they key has the same dimensions but IS NOT the time variable
            if (item.shape == data['time'].shape) & (key != 'time'):
                time_variable_keys += [key]            
            elif key != 'time':
                none_time_variable_keys += [key]
                # data_pro[key] = data[key]

        # prepare out array
        data_pro = {key: np.empty(time_range.shape) for key in time_variable_keys}
        for key in time_variable_keys:
            data_pro[key][:] = np.nan
        data_pro['time'] = time_range
        # just make a copy if its not a time varying key
        for key in none_time_variable_keys:
            data_pro[key] = data[key]
            
        # time matching
        for t_idx, t in enumerate(time_range):
            n_idx = int(binary_search(data['time'], t, lo=max(0, t_idx - 1000),
                                      hi=min(t_idx + 1000, len(data['time']))))
            
            # if the time_range time and data['time'] found in this iteration are within an acceptable range (1 hour)
            t_diff = t - data['time'][n_idx]
            if t_diff.total_seconds() <= 3600.0:
                # only for time varying keys
                for key in time_variable_keys:
                    try:
                        data_pro[key][t_idx] = data[key][n_idx]
                    except:
                        print data.keys()
                        print data_pro.keys()
                        print key
                        print t_idx
                        print n_idx
                        data_pro[key][t_idx] = data[key][n_idx]

        return data_pro

    # find start and end times to time match to based on the two datasets
    start = np.max([xdata['time'][0], ydata['time'][0]])
    end = np.min([xdata['time'][-1], ydata['time'][-1]])
    time_range = date_range(start, end, 60, 'minutes')

    # if the time resolutions of both datasets are not the same, raise an error
    if xdata['time'][1] - xdata['time'][0] != ydata['time'][1] - ydata['time'][0]:
        raise ValueError('xdata time resolution does not match ydata!')

    # time match up the data now a common time line has been made (time_range)
    xdata_pro = time_match_process_single_site(xdata, time_range)

    ydata_pro = time_match_process_single_site(ydata, time_range)

    return xdata_pro, ydata_pro


# plotting

def density_scatter(ax, x, y, s = 10, edgecolor = '',vmin=1e-5, vmax=2e-4, gamma=1. / 2.):

    """
    Plot the scatter data onto the axis, coloured by the 'local' density of points. Code effectively
    makes a 2D pdf of the scatter points, then colours each scatter point according to the PDF value at their point.
    Values are plotted such that the points of highest density are last (ontop). Note: This code is best used on
    'smoothly varying' density plots (so the plotted high density scatter points do not overlay very low density points)

    :param ax: figure axis
    :param x: x data
    :param y: y data
    :param s: size of scatter point
    :param edgecolor: edge colour of scatter point
    :param vmin: minimum value on the density colourbar
    :param vmax: maximum value on the density colourbar
    :param gamma: level of transparency each scatter point will have (so you can see the points behind it)
    :return: scat
    """

    from scipy.stats import gaussian_kde
    import matplotlib.colors as colors

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # scatter the points onto the axis
    scat = ax.scatter(x, y, c=z, s=s, edgecolor=edgecolor,
                            norm=colors.PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))

    return scat

def fig_majorAxis(fig):

    """
    Add an overlying axis to the plot, in order to set figure x and y axis titles.

    :param fig:
    :return: ax
    """

    import matplotlib.pyplot as plt

    ax = fig.add_subplot(111)
    # ax = plt.subplot2grid((1, 1), (0, 0))

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.patch.set_visible(False)

    return ax

def invisible_axis(ax):

    """
    Make axis invisible

    :param ax:
    :return: ax
    """

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.patch.set_visible(False)

    return ax

def linear_fit_plot(x, y, ax, ls = '-', color = 'black'):

    # make sure they are np arrays so idx can be an array of values
    x = np.array(x)
    y = np.array(y)

    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1)

    ax.plot(np.array([-100,100]), m * np.array([-100,100]) + b, ls = ls, color = color)

    return m, b

def add_at(ax, text, loc=2, size=10):

    """
    Add a label to the plot. Used for alphabetising subplots. Automatically label in the top left
    corner

    :param ax:
    :param text: 'string'
    :param loc:
    :return: _at (no idea why the underscore...)
    """

    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

    fp = dict(size=size)
    _at = AnchoredText(text, loc=loc, prop=fp, frameon=False)
    ax.add_artist(_at)
    return _at

def discrete_colour_map(lower_bound, upper_bound, spacing):

    """Create a discrete colour map"""

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # jet by default
    # if cmap == '':
    #     cmap = plt.cm.jet
    cmap = plt.cm.viridis
    cmap = plt.cm.jet

    # extract all colors from the .jet map
    #cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    # cmaplist[0] = (.5, .5, .5, 1.0)
    # create the new map
    cmap22 = cmap.from_list('Custom cmap', cmaplist, cmap2.N)

    # define the bins and normalize
    bounds = np.linspace(lower_bound, upper_bound, spacing)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm

# other

def ensure_dir(file_path):

    """
    create directory if one does not already exist
    http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    copied 24/02/17
    :param file_path:
    :return:
    """

    import os

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline

def add_old_water(data):
    C_qf = 80 # Quick Flow conductivity
    
    #prepare data array
    data['filledCond'] = data.combinedCond.interpolate(method= 'linear')
    data_strp = data.dropna(subset=['filledCond']).copy() #remove leading and trailing gaps
    x = data_strp['filledCond'].to_numpy()
    Q_arr = data_strp['Qcms'].to_numpy()

    # find local min and local max using scipy.signal.find_peaks 
    pks, trs = find_peaks_and_troughs(x)

    # get old water cond.
    C_bf = get_old_water_conductivity(x, pks, trs)

    # Estimate Old/new water fraction and absolute values
    frac_bf = (x-C_qf)/(C_bf-C_qf)
    Q_bf = Q_arr*frac_bf
    Q_qf = Q_arr-Q_bf

    # store in original DF
    data_strp['Cond_bf'] = C_bf
    data_strp['Q_bf'] = Q_bf
    data = data.join(data_strp[['Cond_bf','Q_bf']])
    data.loc[data.combinedCond.isna(),['Cond_bf','Q_bf']] = np.nan #remove interpolated values

    # Old water shouldn't be higher than total Q
    data['Q_bf'] = data['Q_bf'].where((data['Q_bf']<data['Qcms']) | (data['Q_bf'].isnull()) ,
                                  data['Qcms'])

    data['Q_qf'] = data['Qcms'] - data['Q_bf']

    return data

def get_old_water_conductivity(x, pks, trs):

    x_pks = x[pks]
    fip = InterpolatedUnivariateSpline(pks,x_pks,k=1)
    pks_t = fip(range(len(x)))

    x_trs = x[trs]
    fitr = InterpolatedUnivariateSpline(trs,x_trs,k=1)
    trs_t = fitr(range(len(x)))

    # get mean of envelope as the base flow conductivity
    C_bf = (pks_t + trs_t) / 2
    return C_bf

def find_peaks_and_troughs(x):
    #peaks params
    prominence = (6,50)
    width = 7
    wlen = 100    

    #Find non event-related peaks and troughs    
    pks = find_peaks(x, prominence=prominence, width = width, wlen = wlen)[0]
    trs = find_peaks(-x, prominence=prominence, width = width, wlen = wlen)[0]

    #add and remove pks and troughs manually identified
    refined_peaks, refined_trs = refine_peaks_and_troughs(pks, trs)

    return refined_peaks, refined_trs

def refine_peaks_and_troughs(pks, trs):

    pks_append_list = [640, 1115, 1155, 1330, 1342, 1357,1448, 1731,
                    2213, 8560, 9218, 26406, 27828]
    trs_append_list = [590, 1130,1262, 1316,  1335, 1424, 1698, 1793,
                    3135, 3910, 9264,11870, 11917, 12145, 18542,
                    ]

    pks_remove_list = [2033, 13424, 17900, 18048, 19088, 22207, 22357, 22366, 22569,
                    24677, 25415, 27571, 27828, 28050, 28086, 28752, 28754,
                    29071, 30053, 31120, 32257, 32262, 33425, 33526, 
                    33995, 34835, 34858, 34924, 34926, 37064, 39901,
                    40109, 40230, 41242, 43218, 43241, 43763, 43807,
                    46566, 46676, 48016, 52974, 53014, 63978]

    trs_remove_list = [1165, 2001, 2079, 2904, 4039, 10213,12002,
                    13400, 16653, 17989, 18022, 18998, 19119, 19775, 22222, 22339, 22613,
                    25448, 27580, 27859, 28104, 28771, 29087, 30017,
                    30218, 31141, 32118, 32267, 33381, 33382, 33552, 
                    34826, 34844, 34908, 34912, 39892, 41250, 43256,
                    43780, 43822, 46583, 46664, 48027, 51376, 52987,
                    53074, 61450, 61537, 63879, 63992]

    #add and remove pks and trs:  
    pks = np.delete(pks, np.isin(pks,pks_remove_list))
    pks = np.append(pks,pks_append_list)
    pks.sort()
    trs = np.delete(trs, np.isin(trs,trs_remove_list))
    trs = np.append(trs,trs_append_list)
    trs.sort()

    return pks, trs

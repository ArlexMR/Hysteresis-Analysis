import pandas as pd
import numpy as np
import datetime
from src.h_index import get_h_index

def get_event_matrix(Events, data_P, data_full):
    h = get_h_index(Events) # calc h index
    discharge_DF    = get_discharge_metrics(Events, data_full) # Calc discharge related metrics
    turb_DF         = get_turb_metrics(Events)
    load_DF         = get_load_metrics(Events, data_full)
    rainfall_DF     = get_rainfall_metrics(data_P, discharge_DF) # discharge_DF provides the events (times)
    Full_featuresDF = h.join(discharge_DF).join(turb_DF).join(load_DF).join(rainfall_DF)

    # Remove unused columns
    Full_featuresDF.drop(labels = [('Q_Event','TimePeakQ'),('Q_Event','TimePeakQf'), ('Q_Event','TimeStartQ'), ('T_Event','TimePeakT')], axis = 1, inplace = True)
   
    return Full_featuresDF


def get_discharge_metrics(data, data_full):
    EventDF = data.groupby('Event_ID')['Qcms'].max().rename('peakFlow').to_frame().join( #add peakFlow
        data.groupby('Event_ID')['Qcms'].idxmax().rename('TimePeakQ')).join(  # add time of peak flow
        data.groupby('Event_ID')['Q_qf'].idxmax().rename('TimePeakQf')).join(# add time of peak quick flow
        data.reset_index().groupby('Event_ID')['Date'].first().rename('TimeStartQ'))# add starting time
    start_flow = data.groupby('Event_ID')['Qcms'].first() 
    EventDF['Hours2Peak'] = [t.total_seconds()/3600 for t in EventDF['TimePeakQ'].subtract(EventDF['TimeStartQ'])]  # add time to peak
    EventDF['DeltaQ'] = EventDF['peakFlow'] - start_flow 

    # Quick Flow and base flow params:
    EventDF = EventDF.join(data.groupby('Event_ID').apply(dischargeStats,  
                                                colQ = 'Qcms',
                                                colBF = 'Q_bf',
                                                colT = 'combinedTurb',
                                                col_load = 'load_mgs',
                                                max_nan = 1 # nan values already checked
                                                ).droplevel(1,axis = 0))
    EventDF.columns = pd.MultiIndex.from_product([['Q_Event'],EventDF.columns]) # add second column index


    # Discharge antecedent features
    AntecedentDF = data.groupby('Event_ID')['Qcms'].first().rename('startFlow').to_frame()
    cum_disch = pd.DataFrame()
    cum_disch['cum_Qcm_10D'] = data_full.Qcms.multiply(60*15).rolling('10D', min_periods= round(24*4*0.9)).sum()
    cum_disch['cum_Qcm_15D'] = data_full.Qcms.multiply(60*15).rolling('15D', min_periods= round(24*4*0.9)).sum()
    cum_disch['cum_Qcm_20D'] = data_full.Qcms.multiply(60*15).rolling('20D', min_periods= round(24*4*0.9)).sum()
    cum_disch['cum_Qcm_25D'] = data_full.Qcms.multiply(60*15).rolling('25D', min_periods= round(24*4*0.9)).sum()
    cum_disch['cum_Qcm_30D'] = data_full.Qcms.multiply(60*15).rolling('30D', min_periods= round(24*4*0.9)).sum()

    AntecedentDF = AntecedentDF.join( cum_disch.loc[EventDF[('Q_Event','TimeStartQ')],:].set_index(EventDF.index) )
    AntecedentDF.columns = pd.MultiIndex.from_product([['Q_Antecedent'],AntecedentDF.columns]) # add second column index

    Discharge_DF = EventDF.join(AntecedentDF)

    return Discharge_DF

def get_turb_metrics(data):
    # TurbDF = data.groupby('Event_ID')['combinedTurb'].agg(peakTurb = 'max', meanTurb = 'mean') # peak and mean turbidity
    TurbDF = data.groupby('Event_ID').apply(turbStats 
                                            , colQ = 'Qcms'
                                            , colBF = 'Q_bf'
                                            , colT = 'combinedTurb'
                                            , col_load = 'load_mgs'
                                            , max_nan = 1 # nan values already checked
                                            ).droplevel(1, axis = 0)

    TurbDF['TimePeakT'] = data.groupby('Event_ID')['combinedTurb'].idxmax().rename('TimePeakT') # time at peak T
    TurbDF['QatTpeak'] = data.loc[TurbDF['TimePeakT'],'Qcms'].to_frame().set_index(TurbDF.index) # Q at peak T
    
    TurbDF.columns = pd.MultiIndex.from_product([['T_Event'],TurbDF.columns])

    return TurbDF

def get_rainfall_metrics(data_P, Discharge_DF):
    #reference time for P metrics:
    hours_after_Q_peak = 4 #end of event for P event-related mmetrics
    days_before_start = 1 # start of event for P antecedent-related metrics

    # Event and antecedent Parameters in the Precip dataframe:  
    # Event-related Accumulated P 
    data_P['P_10h_acum'] = data_P.precip.rolling(10 + hours_after_Q_peak, min_periods=round(10 * 0.9)).sum()
    data_P['P_11h_acum'] = data_P.precip.rolling(11 + hours_after_Q_peak, min_periods=round(11 * 0.9)).sum()
    data_P['P_12h_acum'] = data_P.precip.rolling(12 + hours_after_Q_peak, min_periods=round(12 * 0.9)).sum()
    data_P['P_13h_acum'] = data_P.precip.rolling(13 + hours_after_Q_peak, min_periods=round(13 * 0.9)).sum()
    data_P['P_14h_acum'] = data_P.precip.rolling(14 + hours_after_Q_peak, min_periods=round(14 * 0.9)).sum()
    data_P['P_15h_acum'] = data_P.precip.rolling(15 + hours_after_Q_peak, min_periods=round(15 * 0.9)).sum()

    # Event-related Intensity
    data_P['P_12h_max_i'] = data_P.precip.rolling('12H').max() #Max intensity value in the period
    data_P['P_15h_max_i'] = data_P.precip.rolling('15H').max() #Max intensity value in the period
    data_P['P_12h_mean_i'] = data_P.precip.replace({0:np.nan}).rolling('12H').mean().replace({np.nan:0}) #mean intensity when rain !=0
    data_P['P_15h_mean_i'] = data_P.precip.replace({0:np.nan}).rolling('15H').mean().replace({np.nan:0}) #mean intensity when rain !=0

    # dates to sample P metrics:
    relevant_dates_event = [t.ceil('H') + datetime.timedelta(hours = hours_after_Q_peak) for t in Discharge_DF[('Q_Event','TimePeakQ')]]
    # relevant_dates = [t.ceil('H') for t in DF_h['StartTimeQ']] 

    P_event_DF = data_P.loc[relevant_dates_event].drop(['HourlyPrecipitation','TFlag','precip','sFlag','missFlag'],axis=1).set_index(Discharge_DF.index)
    P_event_DF.columns = pd.MultiIndex.from_product([['P_event'],P_event_DF.columns])


    # Antecedent P
    anteced_P = pd.DataFrame()

    anteced_P['P_50d_acum'] = data_P.precip.rolling('50D', min_periods= round(50*24 * 0.9)).sum()
    anteced_P['P_40d_acum'] = data_P.precip.rolling('40D', min_periods = round(40*24 * 0.9)).sum()
    anteced_P['P_30d_acum'] = data_P.precip.rolling('30D', min_periods = round(30*24 * 0.9)).sum()
    anteced_P['P_25d_acum'] = data_P.precip.rolling('25D', min_periods = round(25*24 * 0.9)).sum()
    anteced_P['P_20d_acum'] = data_P.precip.rolling('20D', min_periods = round(20*24 * 0.9)).sum()

    # dates to sample P antecedent metrics:
    relevant_dates_anteced = [t.ceil('H') - datetime.timedelta(days = days_before_start) for t in Discharge_DF[('Q_Event','TimeStartQ')]]


    P_antec_DF = anteced_P.loc[relevant_dates_anteced].set_index(Discharge_DF.index)
    P_antec_DF.columns = pd.MultiIndex.from_product([['P_antec'],P_antec_DF.columns])

    P_DF= P_event_DF.join(P_antec_DF)
    
    return P_DF

def get_load_metrics(data, data_full):
    Event_time_start =  data.reset_index().groupby('Event_ID')['Date'].first().rename('TimeStartQ')

    load_df = data.groupby('Event_ID').apply(loadStats 
                                         , colQ = 'Qcms'
                                         , colBF = 'Q_bf'
                                         , colT = 'combinedTurb'
                                         , col_load = 'load_mgs'
                                         , max_nan = 1 # nan values already checked
                                        ).droplevel(1, axis = 0)

    load_df.columns = pd.MultiIndex.from_product([['Load_Event'],load_df.columns])

    #Antecedent Conditions
    # Antecedent conditions
    load = pd.DataFrame()
    load['load_15D'] = data_full.load_mgs.multiply(60*15).rolling('15D', min_periods= round(24*4*0.9)).sum()
    load['load_20D'] = data_full.load_mgs.multiply(60*15).rolling('20D', min_periods= round(24*4*0.9)).sum()
    load['load_25D'] = data_full.load_mgs.multiply(60*15).rolling('25D', min_periods= round(24*4*0.9)).sum()
    load['load_30D'] = data_full.load_mgs.multiply(60*15).rolling('30D', min_periods= round(24*4*0.9)).sum()
    load['load_40D'] = data_full.load_mgs.multiply(60*15).rolling('40D', min_periods= round(24*4*0.9)).sum()

    load_antec = load.loc[Event_time_start,:].set_index(Event_time_start.index)

    load_antec.columns = pd.MultiIndex.from_product([['Load_antec'],load_antec.columns])

    load_df = load_df.join(load_antec)

    return load_df

def dischargeStats(df,colQ, colBF, colT, col_load, max_nan = 0.1):
    full_len = len(df)
    filtDF = df[[colQ, colBF, colT, col_load]].dropna() #filter over all discharge and load variables for consistency in metrics
    newlen = len(filtDF)
    
    filtDF['qf'] = filtDF[colQ] - filtDF[colBF]
    out_df = pd.DataFrame({
                            # total discharge params
                           'meanFlow':[filtDF[colQ].mean()],
                           'totFlow': [filtDF[colQ].sum()*(60*15)],
                           # Quick Flow params
                           'cum_qf': [filtDF['qf'].sum()*(60*15)],
                           'mean_qf': [filtDF['qf'].mean()],
                           'max_qf': [filtDF['qf'].max()],
                            #Base flow params
                           'cum_bf': [filtDF[colBF].sum()*(60*15)],
                           'mean_bf': [filtDF[colBF].mean()],
                           'max_bf': [filtDF[colBF].max()],
                      })
    out_df['bf_qf_ratio']= out_df['cum_bf']/out_df['cum_qf']
    out_df['bf_qf_peak_ratio'] = out_df['max_bf']/out_df['max_qf']
    out_df['Tf_qf_ratio']= out_df['totFlow']/out_df['cum_qf']
    out_df['Tf_qf_peak_ratio'] = filtDF[colQ].max()/out_df['max_qf']
    out_df['Tf_bf_ratio']= out_df['totFlow']/out_df['cum_bf']
    out_df['Tf_bf_peak_ratio'] = filtDF[colQ].max()/out_df['max_bf']
    
    if full_len-newlen > max_nan*full_len:
        out_df.loc[:] = np.nan
    return out_df
 
def loadStats(df,colQ, colBF, colT, col_load, max_nan = 0.1):
    full_len = len(df)
    filtDF = df[[colQ, colBF, colT, col_load]].dropna() #filter over all discharge and load variables for consistency in metrics
    newlen = len(filtDF)
    out_df = pd.DataFrame({
                            #Load params
                           'maxLoad': [filtDF[col_load].max()],
                           'totalLoad': [filtDF[col_load].sum()],
                           'meanLoad': [filtDF[col_load].mean()],
                      })
    
    if full_len-newlen > max_nan*full_len:
        out_df.loc[:] = np.nan
        
    return out_df
 
def turbStats(df,colQ, colBF, colT, col_load, max_nan = 0.1):
    full_len = len(df)
    filtDF = df[[colQ, colBF, colT, col_load]].dropna() #filter over all discharge and load variables for consistency in metrics
    newlen = len(filtDF)
    out_df = pd.DataFrame({
                            #turb params
                           'peakTurb': [filtDF[colT].max()],
                           'meanTurb': [filtDF[colT].mean()],
                      })
    
    if full_len-newlen > max_nan*full_len:
        out_df.loc[:] = np.nan
        
    return out_df
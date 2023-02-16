import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def clean_precip(data_P):

        # Get hourly P by using data at HH:54 or HH:00
        data_P = data_P[(data_P.index.minute==54) | (data_P.index.minute==0)]\
                .resample('H',closed='right',label='right').last()\
                .fillna(value=np.nan)

        # trace precip:
        data_P['TFlag'] = np.nan
        data_P.loc[data_P.HourlyPrecipitation=='T','TFlag'] = 1
        data_P.HourlyPrecipitation.replace({'T':0}, inplace=True)

        # s flags:
        data_P['precip'] = [float( str(x).rstrip('s') ) for x in data_P['HourlyPrecipitation'] ]
        data_P['sFlag']  = [str(x).lstrip('0123456789.') for x in data_P['HourlyPrecipitation']]
        data_P['sFlag'].replace(['', 'nan'], np.nan, inplace = True )
        data_P['sFlag'].replace('s', 1, inplace = True )

        # flag missing values
        data_P['missFlag'] = pd.to_numeric(data_P.HourlyPrecipitation.isna()).replace({False : np.nan, True : 1 })
        data_P = data_P[:'2019-07-03 11:15:00']

        return data_P

def clean_QTC(data_csv):
        # remove turbidity outliers
        filtered_data_csv = remove_outliers(data_csv)

        # interpolate gaps with less than 30 missing values 
        filled_Data = filtered_data_csv.apply(interp_if_gap_greater_than,gap_len=30)
        
        # Unify turbidity 
        combined_turb= combine_turb(filled_Data) 

        # Unify conductivity
        combined_cond = combine_cond(combined_turb)

        # Output
        cleaned_data = combined_cond

        return cleaned_data

def remove_outliers(data_csv):
        eraseDates=['2018/01/19 12:15',
                        '2018/02/07 06:15',
                        '2018/03/25 18:30',
                        '2018/03/25 19:45',
                        '2018/03/25 20:30',
                        '2018/03/26 06:00',
                        '2018/03/26 09:45',
                        '2018/11/06 14:30',
                        '2018/11/06 16:00',
                        '2018/11/06 19:00',
                        '2019/01/14 02:15',
                        '2019/05/04 21:00',
                        '2019/05/04 21:15',
                        '2019/05/04 23:15',
                        '2019/05/05 00:15',
                        '2019/05/05 00:30',
                        '2019/05/05 00:45',
                        '2019/05/05 01:00',
                        '2019/05/07 04:15',
                        '2019/05/07 05:15',
                        '2019/06/21 23:00',
                        '2019/06/26 14:30',
                        '2019/06/27 04:15',
                        '2019/06/27 20:15'
                        ]
        data_csv.loc[eraseDates,['turb_NTU','turbE_NTU']]=np.nan
        data_csv.loc['2019/05/07 23:15':'2019/05/08 04:15',['turb_NTU','turbE_NTU']]=np.nan
        data_csv.loc['2018/12/31 20:30','turb_NTU']=np.nan

        data_csv.loc['2019/03/01 11:00':,'Cond_uscm'] = np.nan 
        data_csv.loc['2019/02/24 02:30':'2019/02/24 12:00','condE_uscm'] = np.nan 
        
        return data_csv

def interp_if_gap_greater_than(ser,gap_len): 
    
        #ser: pandas series to linearly interpolate (doesnot consider index)
        #gap_len: max gap length to interpolate

        df=pd.DataFrame()    
        
        #numbering gap's and nongaps sequences
        df['2']=(ser.isnull().diff()!=0).cumsum() 
        
        # replace numbering by sequence length and discard nongap's:
        df['3']=df.groupby('2')['2'].transform('count')*ser.isnull() 

        df['4']=ser.interpolate() #interpolate the whole series

        df.loc[df['3']>gap_len,'4']=np.nan # replace by nan where gap>gap_len

        return df['4']

def combine_turb(data):
        xy=data[['turb_NTU','turbE_NTU'] ].dropna(axis=0, how='any')
        
        x=np.array(xy.turb_NTU).reshape(-1, 1)
        y=np.array(xy.turbE_NTU).reshape(-1, 1)

        # linear regression
        model = LinearRegression().fit(x, y)

        # y = mx+b
        m = model.coef_
        b = model.intercept_
 
        #Full Series
        data['combinedTurb'] = data.turbE_NTU.combine_first(data.turb_NTU * m[0] + b)
        
        return data

def combine_cond(data):
        xy=data[['Cond_uscm','condE_uscm'] ].dropna(axis=0, how='any')
        y=np.array(xy.condE_uscm).reshape(-1, 1)
        x=np.array(xy.Cond_uscm).reshape(-1, 1)

        # linear regression
        model = LinearRegression().fit(x, y)
        # y = mx+b
        m = model.coef_
        b = model.intercept_

        data['combinedCond']=data.condE_uscm.combine_first(data.Cond_uscm * m[0] + b)

        data['filledCond'] = data.combinedCond.interpolate(method= 'linear')


        return data
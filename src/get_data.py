import pandas as pd


def read_precip(P_Path):
    cols=['DATE','HourlyPrecipitation']
    data_P=pd.read_csv(P_Path,index_col='DATE',parse_dates=True,usecols=cols,encoding='unicode_escape',low_memory=False,dayfirst=True,sep=',')
    return data_P

def read_QTC(QTC_Path):

    cols=['Qcms','Date','Cond_uscm','turb_NTU','condE_uscm','turbE_NTU','depthE_m']
    data_csv=pd.read_csv(QTC_Path,index_col='Date',parse_dates=True,usecols=cols,encoding='unicode_escape',low_memory=False,dayfirst=True)

    #Cast data:
    data_csv=data_csv.apply(pd.to_numeric,errors='coerce')
    return data_csv


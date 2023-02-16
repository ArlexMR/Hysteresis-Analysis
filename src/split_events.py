import pandas as pd
import numpy as np

def split_events(filled_Data):
    n = 0.25 # n day interval
    epsilon = 0 # local min is function of epsilon and is defined  as:   Qi <= Qi-1 - epsilon AND  Qi < Qi+1 + epsilon

    #Get splitting dates
    splitting_date=pd.date_range(start=filled_Data.index[0],end=filled_Data.index[-1],freq=str(n)+'D')

    #get last local min including window limits
    datamin=filled_Data.Qcms.resample(str(n)+'D').agg(get_min)
    datamin.reset_index(level=1,drop=True,inplace=True) #remove second index

    #Set real datetime as index while preserving window index (resample index)
    datamin['DateWindow']=datamin.index
    datamin.set_index('datemin',inplace=True)

    # non mins 
    back=datamin.Qmin.diff().add(epsilon).le(0) # True if Qi<=Qi-1 - epsilon
    forw=datamin.Qmin.diff(-1).subtract(epsilon).lt(0) # True if Qi<Qi+1 + epsilon
    PotentialStart=datamin.loc[back & forw]['Qmin'].copy()
    PotentialStart=PotentialStart.to_frame() #

    # Flagging dataframe with unique Event ID for each potential event
    PotentialStart['EventID']=range(1,len(PotentialStart)+1,1)
    filled_data_extend=filled_Data.join(PotentialStart['EventID'])
    filled_data_extend['EventID']=filled_data_extend['EventID'].fillna(method='ffill')

    # check event conditions for each potential event between local mins 
    check_events=filled_data_extend.dropna(subset=['EventID']).groupby('EventID')['Qcms'].transform(check_Event).rename('Event_checker')

    #Eliminate ID's of non-events
    tmp=filled_data_extend.join(check_events)
    filled_data_extend['EventID_2']=tmp['EventID']*tmp['Event_checker'] # replace ID by nan

    #Forwardfill event ID's
    filled_data_extend['EventID_22']=filled_data_extend['EventID_2'].fillna(method='ffill')
    filled_data_extend.drop(['EventID','EventID_2'],axis=1,inplace=True) #drop temporary ID's
    filled_data_extend.rename(columns={'EventID_22':'Event_ID'},inplace=True) #updating final ID column name

    #renumbering ID's
    currentIDs=filled_data_extend.Event_ID.dropna().unique()
    newID=range(1,len(currentIDs)+1,1)
    filled_data_extend['Event_ID']=filled_data_extend['Event_ID'].map(dict(zip(currentIDs,newID)))

    # Get starting points
    finalStart=filled_data_extend[filled_data_extend.Event_ID.diff()>0]

    # refine events based on manual inspection
    filled_data_extend.loc[filled_data_extend.Event_ID==58,'Event_ID']=57 #Join 57 and 58
    filled_data_extend.loc[filled_data_extend.Event_ID==97,'Event_ID']=96 #IDEM
    filled_data_extend.loc[filled_data_extend.Event_ID==102,'Event_ID']=101
    filled_data_extend.loc[filled_data_extend.Event_ID==113,'Event_ID']=112
    filled_data_extend.loc[filled_data_extend.Event_ID==114,'Event_ID']=112

    # Apply drop tails function
    # newID with dropped non-event rows
    newID=filled_data_extend.groupby('Event_ID').apply(dropTails).reset_index(level=0,drop=True).rename('new_ID')
    filled_data_extend = filled_data_extend.join(newID) # join newID
    filled_data_extend.drop('Event_ID',axis=1,inplace=True) #delete pevious ID
    filled_data_extend.rename({'new_ID':'Event_ID'},axis=1,inplace=True) #rename newID
    
    return filled_data_extend

def filter_events(DF):
    # Manually selected events
    sel_events=[9,14,18,19,23,24,26,35,36,39,43]\
    +[i for i in range(48,58)]\
    +[i for i in range(59,76)]\
    +[i for i in range(85,97)]\
    +[i for i in range(98,102)]\
    +[i for i in range(103,113)]\
    +[119,120,123,124]\
    +[i for i in range(131,140)]

    return DF[DF.Event_ID.isin(sel_events)]

# get (last) min within windows:
def get_min(serie):
    serie=serie[::-1] #reverse order to get the last min
    datemin=serie.idxmin()
    Qmin=serie.loc[datemin]
    return pd.DataFrame({'datemin':[datemin],'Qmin':[Qmin]})

# Check conditions within each event : Qmax>=1.5Qmin and DeltaQ>1
def check_Event(df):
    cond1=(df.max()/df.min()>1.5)
    cond2=((df.max()-df.min())>1)
    if cond1 and cond2:
        return 1
    else:
        return np.nan

# Function to exclude tails from each event
def dropTails(DF):
    peakQidx=DF.Qcms.idxmax() #idx of peak flow
    cumm_Qchange=DF.loc[peakQidx:,'Qcms'].diff().abs().rolling(24).sum() # added change over last 24 / 4/h = 6 h
    Event_end=cumm_Qchange.where(cumm_Qchange==0).first_valid_index() # first cummulative change equal zero 
    if not(Event_end == None): # if found and event end:
        DF.loc[Event_end:,'Event_ID']=np.nan #replace post end data by nan
    return DF.Event_ID


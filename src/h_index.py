import pandas as pd
import numpy as np


# calculating h-index for the whole series
def get_h_index(DF):

    #ID's of (manually) selected events:
    sel_events = DF.Event_ID.unique()

    data = DF.dropna(subset = ['Event_ID']) # remove nan's
    
    # Apply h index calc. function to each Event_ID
    DF_h_qf=data[data['Event_ID'].isin(sel_events)].groupby('Event_ID').apply(apply_h_calc)
    
    # Clean and organize output
    DF_h_qf.reset_index(level=1,drop=True,inplace=True) # drop an irrelevant index
    DF_h_qf.drop(['norm_Q','deltaA'],axis=1,inplace=True) #Drop columns not used
    DF_h_qf.columns = pd.MultiIndex.from_tuples([('Response','h')])

    return DF_h_qf


def apply_h_calc(event):

    norm=normalize(event)

    y_desired=get_y_desired(norm, frac_increase=0.01)

    #interpolate data at y_desired steps
    resampled=resample_y(norm,y_desired,freq='5S')
    #check that every row has only one y_desired asociated
    
    if not resampled[resampled.drop(norm.columns,axis=1).sum(axis=1)>1].empty:
        return pd.DataFrame({'norm_Q':[-9999],'deltaA':[-9999],'h':[-9999]})
        
    # aprox Q values towards those in y_desired 
    resampled['adj_Qcms']=resampled.Qcms.map(lambda x: y_desired[np.argmin(np.abs(np.array(y_desired)-x))])
    # adj_loop=resampled[['adj_Qcms','combinedTurb']]

    # calculate h_index and deltaA's  
    deltaA,h,R,F=h_index(resampled)
    
    return pd.DataFrame({'norm_Q':[list(deltaA.index)],'deltaA':[deltaA.to_list()],'h':[h]})

# function for resampling dataframe based Q bins (y_desired)
def resample_y(df,y_desired,freq='T',y_name='Qcms'): 
    #df is a dataframe with Q and turb columns and timeIndex

    colnames=[str(round(i,3)) for i in y_desired]

    #resample with freq 
    resampled=df.resample(freq).asfreq().interpolate('linear')

    #generate a column for each y_desired which true value coincide with the corresponding y_desired
    real_val=resampled[y_name].copy()
    for i,j in enumerate(y_desired): 
        resampled[colnames[i]]=(real_val-j).multiply((real_val-j).shift(-1)).le(0)

    resampled=resampled[resampled.any(bool_only=True,axis=1)].copy() #filter resample by True values (those close to y_desired)
     
    return resampled

#create a DF with col1 and col2 normalized
def normalize(event,col1='Qcms',col2='combinedTurb'): # event shold be a DF
    mincol1=event[col1].min()
    maxcol1=event[col1].max()
    normalized_col1=(event[col1]-mincol1)/(maxcol1-mincol1)
    
    mincol2=event[col2].min()
    maxcol2=event[col2].max()
    normalized_col2=(event[col2]-mincol2)/(maxcol2-mincol2)
    
    return normalized_col1.to_frame().join(normalized_col2.to_frame())

# function to round to a multiple
def ceil_to_multiple(number, multiple,number_decimals=2):
    return round(multiple * np.ceil(number / multiple),number_decimals)

def floor_to_multiple(number, multiple,number_decimals=2):
    return round(multiple * np.floor(number / multiple),number_decimals)

# get bins for discharge
def get_y_desired(event,y_name='Qcms',frac_increase=0.05):
 
    rising=event.loc[:event[y_name].idxmax()]
    falling=event.loc[event[y_name].idxmax():]
    
    miny=max(rising[y_name].min(),falling[y_name].min())
    maxy=event[y_name].max()
    
    y_desired=np.arange(ceil_to_multiple(miny,frac_increase,2),floor_to_multiple(maxy,frac_increase,2)+frac_increase*0.1,frac_increase)
    return np.round(y_desired,3)

# calculate deltaA's and h index  
def h_index(adj_loop,Q_col='adj_Qcms',Turb_col='combinedTurb'):


    rising_limb = adj_loop.loc[:adj_loop[Q_col].idxmax()]
    rising_unique=rising_limb.groupby(Q_col,as_index=False).max().set_index(Q_col,drop=False) # if repeated Q in rising_limb, take that with the max turb

    falling_limb = adj_loop.loc[adj_loop[Q_col].idxmax():]
    falling_unique=falling_limb.groupby(Q_col,as_index=False).max().set_index(Q_col,drop=False) # if repeated Q in falling_limb, take that with the max turb

    # check that rising and falling limbs has the same number of bins
    if not len(rising_unique)==len(falling_unique):
        print("Rising and falling limb couldn't be pairwised" )
        print(adj_loop.index[0] )
        
        return pd.Series(-9999),-9999,rising_unique,falling_unique

    #integrating using trapezoidal rule
    A_rising=rising_unique.rolling(2).sum()[Turb_col].multiply(0.5).multiply(rising_unique.diff()[Q_col])
    A_falling=falling_unique.rolling(2).sum()[Turb_col].multiply(0.5).multiply(rising_unique.diff()[Q_col])

    deltaA=A_rising-A_falling
    h=deltaA.sum()
    
    return deltaA,h,rising_unique,falling_unique
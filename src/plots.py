import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd


def plot_interpolated_ts(raw_data, clean_data):
    fig2 , ax2 = plt.subplots()

    fig2.subplots_adjust(right=0.85)


    ax2Q=ax2
    ax2T=ax2Q.twinx()
    ax2C=ax2Q.twinx()
    ax2T.spines.right.set_position(("axes", 1.1))

    ax2Q.plot(clean_data.Qcms,'r',label='filled')
    ax2Q.plot(raw_data.Qcms,'grey')
    # ax2Q.plot(raw_data.Qcms.rolling(1000,center=True).mean(),'k')
    ax2Q.set_ylabel('Discharge',color='grey')

    ax2C.plot(clean_data.Cond_uscm,'r')
    # ax2C.plot(clean_data.Cond_uscm.rolling(96,center=True).mean(),'k')
    ax2C.plot(clean_data.condE_uscm,'r')
    #ax2C.plot(clean_data.condE_uscm.rolling(96,center=True).mean(),'k')
    ax2C.set_ylabel('Conductivity',color='orange')

    ax2C.plot(raw_data.Cond_uscm,color='orange')
    ax2C.plot(raw_data.condE_uscm,color='orange')


    ax2T.plot(clean_data.turb_NTU,'-r')
    ax2T.plot(clean_data.turbE_NTU,'-r')
    ax2T.plot(raw_data.turb_NTU,'darkblue')
    ax2T.plot(raw_data.turbE_NTU,'darkblue')
    ax2T.set_ylabel('Turbidity',color='darkblue')

    ax2Q.legend()

#plotting function
def plotQTC(ax,df):
    #Column names in df
    Qcolumn='Qcms'
    Tcolumn=['turb_NTU','turbE_NTU']
    Ccolumn=['Cond_uscm','condE_uscm']
    #plotting colors 
    Qcolor='k'
    Tcolor='darkblue'
    Ccolor='darkorange'
    
    #linewidth
    lwdt=1
    
    #axis
    axQ=ax
    axT=ax.twinx()
    axC=ax.twinx()
    
    #plots
    if not df.empty:
        axT.plot(df[Tcolumn],color=Tcolor,linewidth=lwdt)
        axC.plot(df[Ccolumn],color=Ccolor,linewidth=lwdt)
        axQ.plot(df[Qcolumn],color=Qcolor,linewidth=lwdt)
        #Annotation
        maxQ=df.Qcms.max()
        maxT=df[['turb_NTU','turbE_NTU']].max().max()
        txt=str(df['Event_ID'][0])
        axQ.annotate(xy=(0.5,0.90),text=txt,xycoords='axes fraction',fontsize=11,horizontalalignment='center')
        
        #Limits
        mini=np.nanmax([df[Tcolumn].min().min(),0])
        maxi=np.nanmax([mini+40,df[Tcolumn].max().max()])
        axT.set_ylim(mini,maxi)
        axQ.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        axT.tick_params(axis='y', pad=-23,colors=Tcolor)
        axT.yaxis.set_major_locator(ticker.AutoLocator())

    axT.xaxis.set_visible(False)
    axC.xaxis.set_visible(False) 

    axC.yaxis.set_visible(False)

def plot_hydrographs(filled_data_extend):

    #parameters
    nrows=5
    ncols=5

    nevents=len(filled_data_extend.Event_ID.unique())
    #nevents=5

    nfigs=int(np.ceil(nevents/(nrows*ncols)))
    eventID=1
    for i in range(nfigs):
        fig, ax=plt.subplots(nrows,ncols,figsize=(16,16))
        ax_idx=0
        for j in range(nrows*ncols):
            axs=ax.flatten()[ax_idx]
            df=filled_data_extend[filled_data_extend.Event_ID==eventID]
            plotQTC(axs,df)
            ax_idx+=1
            eventID+=1
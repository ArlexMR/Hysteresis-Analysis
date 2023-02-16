import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_time_series(QTC_clean, precip_clean, Events_ts):

    fig, axs = plt.subplots(2,1, figsize = (20,6))

    full_ts_plot1(axs[0], QTC_clean.Qcms, precip_clean.precip, Events_ts[['Qcms','Event_ID']])

    full_ts_plot2(axs[1], cond = QTC_clean.combinedCond, turb = QTC_clean.combinedTurb)
    fig.set_facecolor('white')
    fig.subplots_adjust(hspace = 0.05)


def full_ts_plot1(ax, Qcms, precip, events_Qcms):

    font_tick_labels = 17
    font_labels = 20
    color_base = ['#335C67', '#FFF3B0', '#E09F3E', '#9E2A2B','#540B0E']

    Q_params = {'color'     : color_base[0],
                'linestyle' : '-',
                'linewidth' : 1,
                'label'     : 'Discharge\n($m^3 s^{-1}$)',
                'alpha'     : 1} 

    P_params = {'color' : 'k', 
                'label' : 'Precipitation\n($mm day^{-1}$)'
                }
    

    ax.plot(Qcms,**Q_params )
    filtered_Qcms = pd.DataFrame(Qcms).join(pd.DataFrame(events_Qcms), rsuffix= '_filt')
    peak_times = events_Qcms.groupby('Event_ID')['Qcms'].idxmax()
    Q_peaks    = Qcms[peak_times]
    
    t_prev = 0
    arrow_props = {'headlength': 3 ,'headwidth':3, 'width': 1, 'shrink': 0.1, 'facecolor': 'k'}

    for i, (Q , t) in enumerate(zip(Q_peaks, peak_times)):
        if i==0 or (t - t_prev).total_seconds()/86400 > 15:
            ax.annotate(str(i+1), xy = (t,Q), xytext = (-5,20), textcoords = 'offset points', arrowprops = arrow_props )
            t_prev = t
    # ax.scatter(peak_times,Q_peaks)
    # filtered_Qcms = full_Q_with_event_flags.Qcms.where(~full_Q_with_event_flags.Qcms.isna(), np.nan)
    # qline = ax.plot(filtered_Qcms.Qcms_filt, color ='#03071e', linewidth = Q_params['linewidth']*2)
    ax.yaxis.label.set_color(Q_params['color'])

    ax.tick_params(axis='y', colors=Q_params['color'])
    ax.tick_params(left = True, labelleft = True , labelbottom = False, bottom = True)
    ax.tick_params(axis='both', which='major', labelsize=font_tick_labels)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 1e6)
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_ylabel(Q_params['label'], y = 0.35, fontsize = font_labels)

    ax2 = ax.twinx()
    ax2.bar(precip.resample('1D').sum().index, precip.resample('1D').sum()*25.4, **P_params)
    ax2.set_ylim(200,0)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.tick_params(axis='both', which='major', labelsize=font_tick_labels)
    ax2.set_ylabel(P_params['label'], loc = 'top', fontsize = font_labels)



def full_ts_plot2(ax3, cond, turb):
    font_tick_labels = 17
    font_labels = 20
    color_base = ['#335C67', '#FFF3B0', '#E09F3E', '#9E2A2B','#540B0E']
# Conductivity
    cond_params = {'color'     : color_base[2],
                   'linestyle' : '-',
                   'linewidth' : 1,
                   'label'     : 'Conductivity\n($\mu S cm^{-1}$)'
                   }

    turb_params = {'color'     : color_base[3],
                   'linestyle' : '-',
                   'linewidth' : 1, 
                   'label'     : 'Turbidity\n(FTU)'
                   }

    ax2 = ax3.twinx()
    ax2.plot(cond, **cond_params)
    ax2.yaxis.label.set_color(cond_params['color'])
    ax2.yaxis.label.set_fontsize(font_labels)
    ax2.tick_params(axis='y', colors=cond_params['color'])


    ax2.set_ylim(-1000,1000)
    ax2.set_yticks(range(0,1000,300))
    ax2.spines.right.set_bounds(0,900)
    ax2.tick_params(axis='both', which='major', labelsize=font_tick_labels)
    ax2.set_ylabel(cond_params['label'], y = 0.75, fontsize = font_labels)

    # turbidity
    turb[turb <= 0] = 0.001 
    ax3.plot(turb, **turb_params)

    ax3.yaxis.label.set_color(turb_params['color'])
    ax3.yaxis.label.set_fontsize(font_labels)
    ax3.tick_params(axis='y', colors=turb_params['color'])
    ax3.set_yscale('log')
    ax3.set_ylim(1,1100000)
    ax3.set_yticks([1, 10, 100, 1000])
    # ax3.spines.right.set_bounds(1,1000)
    ax3.tick_params(axis='both', which='major', labelsize=font_tick_labels)
    ax3.set_ylabel(turb_params['label'], y = 0.27, fontsize = font_labels)
    
    

    # fig.legend(loc = 'upper center', ncol = 4, bbox_to_anchor = (0.5,1.05), fontsize = font_tick_labels)
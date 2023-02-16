import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_all_models(ax, all_models_dict, models):
    shifts  = [-.15, 0, .15]
    colors = ['r','g','b']

    for color, shift, model_name in zip(colors, shifts, models):
        
        model_dict = all_models_dict[model_name]
        efs_full = pd.DataFrame(model_dict).T

        ax.scatter(efs_full.n_vars + shift, 
                   -efs_full.avg_score, 
                   label = model_name, 
                   alpha = 0.35, 
                   color = color
                   )
        # ax.legend()


def plot_optimum_models(ax, var_selection_summary_DF, models):
    labelfontsize = 17
    tick_label_font = 15
    shifts  = [-.15, 0, .15]
    colors = ['r','g','b']

    for shift, color, model_name in zip(shifts, colors, models):
        subDF = var_selection_summary_DF.loc[var_selection_summary_DF.model == model_name]
        ax.plot(subDF.n_vars + shift, 
                -subDF.avg_score, 
                '.-' ,
                label = model_name, 
                markersize = 15, 
                color = color,
                linewidth = 2.5
                )
        
        # ax.errorbar(subDF.n_vars + shift, 
        #             -subDF.avg_score, 
        #             1.96*subDF.std_dev/(np.sqrt(5)), 
        #             color = color, 
        #             alpha = 0.5,
        #             capsize = 5  
        #             )
    ax.set_xlabel('Number of predictors', fontsize = labelfontsize)
    ax.set_ylabel('$RMSE$', fontsize = labelfontsize)
    
    plt.legend() 

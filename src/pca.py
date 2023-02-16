#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
# np.random.seed(2)

# x1 = np.linspace(0,1)
# x2 = x1 + 0.5*np.random.rand(len(x1))
# x3 = 0.2*np.random.rand(len(x1)) 
# X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1), x3.reshape(-1,1)))
# scaler = preprocessing.StandardScaler().fit(X)
# X_scaled = scaler.transform(X)
# h = np.linspace(0,5)
# pca = PCA()
# pca.fit(X_scaled)

def plot_pca(predictors, response, subset_predictors_pca = None, subset_predictors_plot = None, arrow_colors = 'k', score_cmap = 'cividis'):
    if subset_predictors_pca is None:
        subset_predictors_pca = predictors.columns

    if subset_predictors_plot is None:
        subset_predictors_plot = subset_predictors_pca

    if isinstance(arrow_colors, str):
        arrow_colors = [arrow_colors]*len(subset_predictors_plot)


    X = predictors[subset_predictors_pca]
    
    scaler = preprocessing.StandardScaler().fit(X)

    X_scaled = scaler.transform(X)
    pca = PCA(n_components = 2)
    pca.fit(X_scaled)

    var_idx = [list(X.columns).index(var) for var in subset_predictors_plot]

    biplot(pca, X_scaled, h = response, 
           var_idx = var_idx, 
           arrow_colors = arrow_colors)

def biplot(pca, X_scaled, h, var_idx = None, components = (0,1), 
            cmap = 'cividis', arrow_colors = None, var_labels = None):

    if var_idx is None:
        var_idx = range(len(X_scaled.shape[1]))


    fig = plt.figure(figsize = (5,5))

    ax, sc = plot_scores(fig, pca, X_scaled, h, components = components, 
                         cmap = cmap)

    plot_loadings(pca, ax, components = components, colors = arrow_colors, 
                  var_idx = var_idx, labels = var_labels)

    # fig.legend()
    return




def plot_scores(fig, pca, X_scaled, h , components = (0,1), cmap = 'cividis'):

    scores   = pca.transform(X_scaled) # transformed data points

    ax1 = fig.add_subplot(1,1,1)

    sc = ax1.scatter(scores[:,components[0]], scores[:,components[1]],
                    c = h, cmap = cmap, vmin = -0.25, vmax = 0.25
                    )

    lowlim, maxlim = np.min((ax1.get_xlim()[0], ax1.get_ylim()[0])), np.max((ax1.get_xlim()[1], ax1.get_ylim()[1])) 
    

    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-10, 10)
    explained_variance_ratio = [pca.explained_variance_ratio_[comp] for comp in components]  
    ax1.set_xlabel(f'Scores - PC{components[0]+1} ({round(100*explained_variance_ratio[0])}%)')
    ax1.set_ylabel(f'Scores - PC{components[1]+1} ({round(100*explained_variance_ratio[1])}%)')

    
    return (ax1, sc)

def plot_loadings(pca, ax, components = (0,1), colors = None, var_idx = None, labels = None ):
    loadings = pca.components_             # variable vectors (ncomponents x n_variables)
    if var_idx is None:
        var_idx = range(loadings.shape[1])
    if colors is None:
        colors = ['r']*len(var_idx)
    if labels is None:
        labels = [str(i) for i in var_idx]

    fig = ax.get_figure()
    bbox = ax.get_position()
    l,b,w,h = bbox.x0, bbox.y0, bbox.width, bbox.height
    ax2 = fig.add_axes(rect = [l,b,w,h])
    ax2.patch.set_alpha(0.0)
    plot_circle(ax2)

    for i, var in enumerate(var_idx):
        coeffs = loadings[components,var]
        ax2.arrow(0, 0, coeffs[0], coeffs[1], color = colors[i], alpha = 0.8, width=0.005, head_width = 0.05)


    ax2.set_ylim(-1.05, 1.05)

    ax2.set_xlim(-1.05, 1.05)

    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, left = False, labelleft = False, right = True, labelright = True)
    ax2.set_xlabel(f"Loadings PC{components[0]+1}")
    ax2.set_ylabel(f"Loadings PC{components[1]+1}")
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    

def plot_circle(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x=np.linspace(start=-1,stop=1,num=500)

    y_positive=lambda x: np.sqrt(1-x**2) 
    y_negative=lambda x: -np.sqrt(1-x**2)

    ax.plot(x,list(map(y_positive, x)), color='grey',alpha=0.5)
    ax.plot(x,list(map(y_negative, x)), color='grey',alpha=0.5)
        
    ax.axhline(0, color = 'grey', alpha = 0.5)
    ax.axvline(0, color = 'grey', alpha = 0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



#%%






# #%%

# print('explained_variance_ratio_: ', pca.explained_variance_ratio_, end = '\n\n')
# print('components_: ','\n' ,pca.components_, '\n')
# print('singular_values_: ', pca.singular_values_)

# X_scaled[0,:] = [1.5, 1, 0]
# X_scaled[-1,:] = [-1.5, -0.5, 0]


# x_transformed = pca.transform(X_scaled)
# coeff = pca.components_

# print('X red transformed: ', pca.transform(X_scaled[5,:].reshape(1,-1)))
# print('X red manually transformed: ', X_scaled[5,:]@pca.components_.T )


# fig, axs = plt.subplots(1,2, figsize = (10,5))

# ax1 = axs[0]
# ax2 = axs[1]
# ax1.plot(X_scaled[:,0],X_scaled[:,1],'.')
# ax1.set_aspect('equal')
# ax1.set_xlim(-2,2)
# ax1.set_ylim(-2,2)
# ax1.plot(X_scaled[0,0],X_scaled[0,1],'or')
# ax1.plot(X_scaled[-1,0],X_scaled[-1,1],'og')


# xs = x_transformed[:,0]
# ys = x_transformed[:,1]
# # scalex = 1/(np.max(xs) - np.min(xs))
# # scaley = 1/(np.max(ys) - np.min(ys))
# scalex = 1
# scaley = 1

# ax2.plot(xs * scalex, ys * scaley,'.')
# ax2.plot((xs * scalex)[0], (ys * scaley)[0],'or')
# ax2.plot((xs * scalex)[-1], (ys * scaley)[-1],'og')

# ax2.arrow(0,0, coeff[0,0], coeff[1,0], color = 'b')
# ax2.arrow(0,0, coeff[0,1], coeff[1,1], color = 'c')
# ax2.arrow(0,0, coeff[0,2], coeff[1,2], color = 'k')


# # %%

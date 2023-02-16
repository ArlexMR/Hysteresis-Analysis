import ast
import configparser
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


def apply_var_selection(model_list, X_train, y_train, var_names, n_vars_to_select_in_backward):

    # Get dict of variables using backward selection
    selected_variables_dict = apply_backward_selection(model_list, X_train, y_train, var_names, n_vars_to_select_in_backward)

    # using the backward-selected set for each model, apply exhaustive selection
    selected_vars_list = list(selected_variables_dict.values())

    efs_dict = {}
    for model, selected_vars in zip(model_list, selected_vars_list): 
        
        # Get subset of input variables for this model
        variable_mask    = [bool(var in selected_vars) for var in var_names]
        filtered_X_train = X_train[:, variable_mask]
        
        metric_dict = exhaustive_selector(model, filtered_X_train, y_train, var_names = selected_vars, max_features = n_vars_to_select_in_backward)

        # compiling dict for saving results 
        efs_dict[model.__class__.__name__] = metric_dict

    efs_DF_optimums = efs_dict_to_DF(efs_dict)

    return efs_DF_optimums, efs_dict

def get_var_selection_table(var_selection_DF):
    
    backward_variables = var_selection_DF.feature_names.explode().unique()
    
    backward_n_vars    = len(backward_variables)
    
    variables_matrix   = pd.DataFrame(index = backward_variables)

    for nvar in range(1, backward_n_vars + 1):

        optimum_vars = var_selection_DF.loc[var_selection_DF.n_vars == nvar, 'feature_names'].values[0]

        variables_matrix['n_vars_' + str(nvar)] = [bool(var in optimum_vars) for var in backward_variables]

    return variables_matrix


def efs_dict_to_DF(var_selection_dict):

    optim_metric_DF = pd.DataFrame(columns = ['avg_score','std_dev', 'feature_names','n_vars', 'model'])

    for model_name, metric_dict in var_selection_dict.items():

        metric_dict_as_DF                = pd.DataFrame.from_dict(metric_dict).T[['avg_score','std_dev','feature_names','n_vars']]
        metric_dict_as_DF['model']       = model_name
        metric_dict_as_DF['avg_score']   = pd.to_numeric(metric_dict_as_DF.avg_score)
        metric_dict_as_DF['std_dev']     = pd.to_numeric(metric_dict_as_DF.std_dev)
        rows_optim_vars                  = metric_dict_as_DF.groupby('n_vars')['avg_score'].idxmax()
        optimum_DF                       = metric_dict_as_DF.loc[rows_optim_vars,:]

        optim_metric_DF = pd.concat([optim_metric_DF, optimum_DF])


    return optim_metric_DF

def exhaustive_selector(model, X_train, y_train, var_names, max_features):

    efs  = EFS(model, 
               min_features   = 1,
               max_features   = max_features,
               scoring        = 'neg_root_mean_squared_error',
               print_progress = True,
               cv             = 5,
                )

    efs1 = efs.fit(X_train, y_train)

    metric_dict = efs1.get_metric_dict()

    for fitted_model in metric_dict.keys(): 
        
        metric_dict[fitted_model]['n_vars'] = len(metric_dict[fitted_model]['feature_idx'])
        metric_dict[fitted_model]['feature_names'] = [var_names[i] for i in metric_dict[fitted_model]['feature_idx']]

    return metric_dict

def apply_backward_selection(model_list, X_train, y_train, var_names, n_vars_to_select_in_backward):

    selected_variables_dict = {}
    for model in model_list:
        FS = SequentialFeatureSelector(model,  
                                        n_features_to_select = n_vars_to_select_in_backward, 
                                        direction = 'backward', 
                                        scoring = 'neg_root_mean_squared_error',
                                        n_jobs = -1
                                    )

        FS.fit(X_train, y_train)

        selection_filter = FS.get_support()
        selected_vars = [var for (var, selected) in zip(var_names, selection_filter) if selected]

        model_name = model.__class__.__name__
        selected_variables_dict[model_name] = selected_vars
    
    return selected_variables_dict

def prepare_model_data(data_file, test_size):

    Full_Data = pd.read_csv(data_file,
                            header = [0,1],
                            index_col = [0]
                            )
    # remove multi-index
    Full_Data.columns = Full_Data.columns.get_level_values(1)

    #Divide X and y data
    Full_Data.dropna(inplace=True) #delete row with nan
    X = Full_Data.drop('h', axis = 1)  # independent variables
    y = Full_Data['h']  # dependent variable

    # Scale data
    X_train_pre, X_test_pre, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)
    scaler = preprocessing.StandardScaler().fit(X_train_pre)
    X_train = scaler.transform(X_train_pre)
    X_test = scaler.transform(X_test_pre)

    var_names = list(X.columns)

    return X_train, X_test, y_train, y_test, var_names

def initiate_models():
    config = configparser.ConfigParser()
    config.read('model_params.txt')

    RF_params = dict(config['Random Forest'])
    RF_kwargs = {key: ast.literal_eval(val) for key, val in RF_params.items()}

    KNN_params = dict(config['KNN'])
    KNN_kwargs = {key: ast.literal_eval(val) for key, val in KNN_params.items()} 

    GBT_params = dict(config['GBT'])
    GBT_kwargs = {key: ast.literal_eval(val) for key, val in GBT_params.items()}

    RF = RandomForestRegressor(**RF_kwargs)

    KNN = KNeighborsRegressor(**KNN_kwargs)

    GBT = GradientBoostingRegressor(**GBT_kwargs)

    return [RF, KNN, GBT]


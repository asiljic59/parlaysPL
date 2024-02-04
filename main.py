import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import final_dataset
from helpers import *
def check_regressions(x,y):
    # missing values are MatchRating,HomeWin%,AwayWin%,Draw%
    '''
    data['MatchRating'] = data['MatchRating'].interpolate(method='spline', order=3, limit_direction='both')
    data['HomeWin%'] = data['HomeWin%'].interpolate(method='spline', order=3, limit_direction='both')
    data['AwayWin%'] = data['AwayWin%'].interpolate(method='spline', order=3, limit_direction='both')
    data['Draw%'] = data['Draw%'].interpolate(method='spline', order=3, limit_direction='both')
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # rf_regressor = RandomForestRegressor(n_estimators=100,)
    # Lasso Regression
    lasso_model = Lasso(alpha=0.5)
    lasso_model.fit(x_train, y_train)
    #check_for_outliers(x_train)
    #lasso_assumptions = are_assumptions_satisfied(lasso_model,x_train,y_train)



    y_pred = lasso_model.predict(x_test)

    # Calculate R^2 score
    r2 = get_rsquared(lasso_model,x_test,y_test)
    print(f"Lasso r2 :{r2}")

    # RIDGE MODEL TEST
    ridge_model = Ridge(alpha=0.5)
    ridge_model.fit(x_train,y_train)
    r2_ridge = get_rsquared(ridge_model,x_test,y_test)
    print(f"Ridge r2 :{r2_ridge}")

    #ELASTIC MODEL TEST
    elastic_net_model = ElasticNet(alpha=0.5)
    elastic_net_model.fit(x_train,y_train)
    r2_elastic_net = get_rsquared(elastic_net_model,x_test,y_test)
    print(f"Elastic net r2 :{r2_elastic_net}")


    '''
    
    #RANDOM FOREST REGRESSIOn!
    param_grid = {
        'n_estimators': [100,200,300],
        'max_depth' : [3,4,5],
        'min_samples_split' : [2,4,5]
    }
    rf_model = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf_model,param_grid=param_grid,cv=5,scoring='r2',verbose=2,n_jobs=-1)

    grid_search.fit(x_train,y_train)
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    r2_rf = get_rsquared(best_estimator,x_test,y_test)
    print(f"Random forest r2 :{r2_rf}")
    '''

    """
    #ROBUST REGRESSION
    robust_regression = HuberRegressor(epsilon=6.6) #epsilon is tuning parameter
    robust_regression.fit(x_train,y_train)

    r2_robust = get_rsquared_adj(robust_regression,x_test,y_test)

    print(f"Elastic net r2 :{r2_robust}")
    """
def check_poynomial(x,y):
    # Specify the degree of the polynomial
    degree = 2 # You can adjust this based on your data
    # Transform the input data to include polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)


    check_regressions(x_poly,y)

if __name__ == '__main__':
    filename = 'data/final_data.csv'
    data = pd.read_csv(filename)


    x = data['MatchRating'].values.reshape(-1,1)
    y_home = data['HomeWin%']
    y_draw = data['Draw%']
    y_away = data['AwayWin%']

    #HOME PRECENTAGE
    check_regressions(x,y_home)

    #DRAW PRECENTAGE
    check_regressions(x,y_draw)
    check_poynomial(x,y_draw)

    #AWAY PRECENTAGE
    check_regressions(x,y_away)

    final_data = pd.DataFrame(columns=['HomeTeam','AwayTeam', 'HomeWin%',
                                       'Draw%', 'AwayWin%'])
    X = data['MatchRating'].values.reshape(-1,1)
    Y = data.drop(columns = ['MatchRating'])
    final_model_home = Lasso(alpha=0.1) #0.72
    final_model_home.fit(x,y_home)
    final_model_away = Lasso(alpha=0.1) #~0.15
    final_model_away.fit(x,y_away)
    final_model_draw = Lasso(alpha=0.1) #0.52
    final_model_draw.fit(x,y_draw)

    current_season = pd.read_csv('data/2023-24.csv')
    current_teams = current_season['Home Team']
    current_teams = current_teams.drop_duplicates()
    all_teams = current_teams.values.tolist()


    predictions_home = pd.DataFrame(final_model_home.predict(x))
    predictions_away = pd.DataFrame(final_model_away.predict(x))
    predictions_draw = pd.DataFrame(final_model_draw.predict(x))

    #predictions_home = predictions_home.T
    #predictions_away = predictions_away.T
    #predictions_draw = predictions_draw.T

    final_data.reset_index(drop=True,inplace=True)
    for index,row in current_season.iterrows():
        if row['Round Number'] == 6:
            pass
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_gd = final_dataset.get_goal_difference(home_team,row['Round Number'],current_season)
        away_gd = final_dataset.get_goal_difference(away_team,row['Round Number'],current_season)
        goal_difference = home_gd - away_gd




   # prediction_data.to_csv('data/final_predictions.csv',index=False)








# See PyCharm help at https://www.jetbrains.com/help/pycharm/

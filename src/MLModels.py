import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor as DTR
import example_web as ew

# setting of the input file, preprocessing method and specific model
# predictions_dir = '../predictions/predictions_20_22_test'
# models_dir = '../models/models_20_22_test'
# accuracies_dir = '../accuracies/accuracies_20_22_test'
predictions_dir = '../tests'
models_dir = '../tests'
accuracies_dir = '../tests'
File_Path_Train = '../data/processing/non_time.csv'  # 'non_time_pca.csv' 'non_time.csv'
transform = 'standard'  # 'standard', 'minmax'
scaler = ew.get_scaler(transform)
Model_Name = 'SVM'  # 'LR' 'RF' 'SVM' 'DTR' 'GBR' 'Light' 'Ridge' 'Lasso' 'SGD'
Type = 'feature_selection'  # 'feature_selection', 'PCA'
year_to_be_test = '2020|2021|2022'
target_col = 'PLUS_MINUS'
col_to_drop = ['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'TEAM_ID', 'OPPO_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY',
               'PLAYER_ID', 'PLAYER_NAME', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
               'FT_PCT', 'DREB', 'REB', 'AST', 'PTS', 'PLUS_MINUS']
# read data
X_train, y_train, X_test, y_test = ew.read_data_select_year_test_MLmodels(train_path=File_Path_Train, year_to_be_test=year_to_be_test,
                                            target_col=target_col, col_to_drop=col_to_drop, scaler=scaler, data_pre=Type)



# build models
scores = ['neg_mean_absolute_error']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = 0

    # cross validation for hyper-paramters optimal based on each model
    if Model_Name == 'SVM':
        SVR_params_nonlinear = {
            # randomly sample numbers from 4 to 204 estimators
            'kernel': ['rbf', 'sigmoid', 'poly'],
            # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
            'gamma': ['scale','auto'],
            # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
            'C': [0.1, 1, 10, 100]
        }
        SVR_params_linear = {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100]
        }
        SVR_params = [SVR_params_nonlinear, SVR_params_linear]
        clf = GridSearchCV(SVR(), SVR_params_linear, scoring=score, verbose=10, cv=3, n_jobs=-1)  # default 5 folds CV
    elif Model_Name == 'DTR':
        DTR_params = {'splitter': ['best', 'random'],
                      "max_depth": list(range(30, X_train.shape[1], 30)),
                      'min_samples_split': list(range(2, 39, 6)),
                      'min_samples_leaf': list(range(1, 21, 3)),
                      'min_weight_fraction_leaf': [0.1, 0.3, 0.5],
                      'max_features': ['auto', 'sqrt', 'log2', None],
                      'min_impurity_decrease': [0.1, 0.3, 0.6, 0.9]
                      }
        clf = RandomizedSearchCV(estimator=DTR(), param_distributions=DTR_params,
                                 scoring=score, n_iter=300, cv=5, random_state=1,
                                 verbose=1, n_jobs=-1)
    elif Model_Name == 'RF':
        RFR_params = {'n_estimators': list(range(3, 100, 15)),
                      "max_depth": list(range(30, X_train.shape[1], 30)),
                      'min_samples_split': list(range(2, 39, 6)),
                      'min_samples_leaf': list(range(1, 11, 3)),
                      'max_features': list(range(5, 40, 5)),
                      'min_impurity_decrease': [0.1, 0.3, 0.6, 0.9]}
        clf = RandomizedSearchCV(RFR(), RFR_params,
                                 scoring=score, n_iter=100, cv=3, random_state=1, verbose=10,
                                 n_jobs=-1)  # default 5 folds CV
    elif Model_Name == 'LR':
        tuned_parameters = [{'fit_intercept': [True, False], 'positive': [True, False]}]
        clf = GridSearchCV(LinearRegression(n_jobs=-1), tuned_parameters, scoring=score,
                           n_jobs=-1)  # default 5 folds CV
    elif Model_Name == 'GBR':
        GBR_params = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'learning_rate': [0.1, 0.3, 0.6, 0.9],
                      'n_estimators': list(range(3, 200, 10)),
                      'subsample': [0.1, 0.3, 0.6, 0.9],
                      'alpha': [0.1, 0.3, 0.6, 0.9],
                      "max_depth": list(range(3, 100, 10)),
                      'min_samples_split': list(range(2, 100, 10)),
                      'min_samples_leaf': list(range(1, 100, 10)),
                      'max_features': list(range(5, 40, 5)),
                      'min_impurity_decrease': [0.1, 0.3, 0.6, 0.9]}
        clf = RandomizedSearchCV(estimator=GBR(), param_distributions=GBR_params,
                                 scoring=score,
                                 n_iter=100, cv=3, random_state=1,
                                 verbose=10, n_jobs=-1)
    elif Model_Name == 'Light':
        LightGBM = lgb.LGBMRegressor(bagging_freq=1)
        LightGBM_params = {'objective': ['regression', 'regression_l1', 'huber'],
                           'boosting': ['gbdt', 'dart', 'rf'],
                           'feature_fraction': [0.1, 0.3, 0.6, 0.9],
                           'subsample': [0.1, 0.3, 0.6, 0.9],
                           'num_leaves': list(range(2, 100, 10)),
                           'learning_rate': [0.1, 0.3, 0.6, 0.9],
                           "max_depth": list(range(3, 100, 10)),
                           'n_estimators': list(range(3, 100)),
                           'min_data_in_leaf': list(range(1, 100, 10)),
                           }

        clf = RandomizedSearchCV(estimator=LightGBM, param_distributions=LightGBM_params,
                                 scoring=score,
                                 n_iter=100, cv=5, random_state=1,
                                 verbose=1, n_jobs=-1)
    elif Model_Name == 'SGD':
        SGD = SGDRegressor()
        SGD_params = {
        'penalty': ['l1', 'l2'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
        clf = GridSearchCV(SGD, SGD_params, scoring=score,
                           n_jobs=-1)
    elif Model_Name == 'Ridge':
        Ridge = Ridge()
        Ridge_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        }
        clf = GridSearchCV(Ridge, Ridge_params, scoring=score, verbose=10, n_jobs=-1)
    elif Model_Name == 'Lasso':
        Lasso = Lasso()
        Lasso_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
        }
        clf = GridSearchCV(Lasso, Lasso_params, scoring=score,
                           n_jobs=-1)
    # train model
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Best model performance report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

    # get accuracy measures
    y_true, y_pred = y_test, clf.predict(X_test)
    print('rooted_mean_squared_error:' + str(mean_squared_error(y_true, y_pred, squared=False)))
    print('max_error:' + str(max_error(y_true, y_pred)))
    print('mean_absolute_error:' + str(mean_absolute_error(y_true, y_pred)))
    print('r2_score:' + str(r2_score(y_true, y_pred)))

    # save train and test errors
    accuracies_to_save = ['best_score_:', str(clf.best_score_), 'best_params_:', str(clf.best_params_),
                          'rooted_mean_squared_error:' + str(mean_squared_error(y_true, y_pred, squared=False)),
                          'max_error:' + str(max_error(y_true, y_pred)), 'mean_absolute_error:' + str(mean_absolute_error(y_true, y_pred)),
                          'r2_score:' + str(r2_score(y_true, y_pred))]
    with open(f'{accuracies_dir}/{Type}_{score}_{Model_Name}_{transform}_accuracy.txt', 'w') as f:
        f.writelines(accuracies_to_save)

    # save models and predictions
    dump(clf, f'{models_dir}/{Type}_{score}_{Model_Name}_{transform}_model.joblib')  # save model
    # clf = load('filename.joblib') load model
    print()
    # y_pred to csv
    resultCSVPath = f'{predictions_dir}/{Type}_{score}_{Model_Name}_{transform}_prediction.csv'
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(resultCSVPath, index=False, na_rep=0)

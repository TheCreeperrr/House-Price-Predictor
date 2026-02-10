from sklearn import datasets, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBRegressor
from sklearn.svm import SVR

#Variables
optimize = False
rand = 42

#Fetch the dataset
#Housing: Raw dataset of 20k items
#X: DataFrame containing data in rows and columns
#Y: contains prices. Its rows match with the rows in X.
#Split Dataset into Training and Testing.
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
Y = pd.Series(housing.target, name='Price')
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=rand)


#   MODEL TRAINING FUNCTIONS
#   FOR NON-OPTIMIZED MODELS:
#   1) Create the model according to its type
#   2) Train the model on the training data sets
#   3) Create a prediction of Y_test using X_test.
#   4) Prediction will be compared with Y_test to determine its R2 Score.

def train_linear_regression():
    noobmodel = LinearRegression()
    noobmodel.fit(X_train, Y_train)
    prediction = noobmodel.predict(X_test)
    return prediction


def train_random_forest():
    if optimize:
        forestmodel = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=rand,
            n_jobs=-1
        )
    else:
        forestmodel = RandomForestRegressor(random_state=rand)

    forestmodel.fit(X_train, Y_train)
    prediction = forestmodel.predict(X_test)
    return prediction


def train_gradient_boosting():
    model = GradientBoostingRegressor()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    return prediction


def train_xgboost():
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    return prediction

#SVR is absolute trash without scaling it. So scaling it is done by default and not specific to just the optimization bit.
#R2 score without scaling will be negative which is horrible.
def train_svr():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVR()
    model.fit(X_train_scaled, Y_train)
    prediction = model.predict(X_test_scaled)
    return prediction


def main():
    print("\033[95mYou're running train_funcs.py...")
    print("Try running main.py this time :)\033[0m\n")


if __name__=='__main__':
    main()
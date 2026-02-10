import train_funcs as tf

#Dictionary containing all models along with their training function in train_funcs.
models = {
        'Linear Regression Model': tf.train_linear_regression,
        'Random Forest Model': tf.train_random_forest,
        'Gradient Boosting Model': tf.train_gradient_boosting,
        'XG Boost Model': tf.train_xgboost,
        'Support Vector Regression Model': tf.train_svr
    }


#Colors to make the terminal look cooler :)
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
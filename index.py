import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
dc_listings = pd.read_csv('listings.csv')
# print(dc_listings.shape)
# dc_listings.head()

our_acc_value = 3
dc_listings['distance'] = np.abs(dc_listings.accommodates - our_acc_value)
print(dc_listings.distance.value_counts().sort_index())
dc_listings = dc_listings.sample(frac=1, random_state=0)
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.price.head())
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)
mean_price = dc_listings.price.iloc[:5].mean()
print(mean_price)

dc_listings.drop('distance', axis=1)
train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

def predict_price(new_listing_value, feature_column):
  temp_df = train_df
  temp_df['distance'] = np.abs(dc_listings[feature_column] - new_listing_value)
  temp_df = temp_df.sort_values('distance')
  knn_5 = temp_df.price.iloc[:7]
  predicted_price = knn_5.mean()
  return(predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price, feature_column='accommodates')
print(test_df['predicted_price'])
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**2
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)

norm_train_df = dc_listings.copy().iloc[:2792]
norm_test_df = dc_listings.copy().iloc[2792:]

def predict_price_multivariate(new_listing_value,feature_columns):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing_value[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

cols = ['accommodates', 'bedrooms']
norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate,feature_columns=cols,axis=1)
norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - norm_test_df['price'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)


knn = KNeighborsRegressor(algorithm='brute')
knn.fit(norm_train_df[cols], norm_train_df['price'])
two_features_predictions = knn.predict(norm_test_df[cols])

two_feature_mse = mean_squared_error(norm_test_df['price'], two_features_predictions)
 
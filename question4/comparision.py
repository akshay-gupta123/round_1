import numpy as np

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import polynomial_regression
import lasso_regression

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
#print(y_poly_pred)
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print("Root mean square error for imported Polynomial regression = " + str(rmse))

x_poly = polynomial_regression.polynomial_features(x,degree=2)
model = polynomial_regression.PolynomialRegression()
model.fit(x_poly,y)
y_poly_pred = model.predict(X=x_poly)
#print(y_poly_pred)
rmse2 = np.sqrt(mean_squared_error(y,y_poly_pred))
print("Root mean square error for Polynomial regression from scratch = " + str(rmse2))


"""##############   LASSO REGRESSION    ##################"""

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)
model = Lasso()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
#print(y_poly_pred)
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
print("Root mean square error for imported Lasso regression = " + str(rmse))



x_poly = lasso_regression.polynomial_features(x,degree=2)
model = lasso_regression.LassoRegression(1)
model.fit(x_poly,y)
y_poly_pred = model.predict(X=x_poly)
#print(y_poly_pred)
rmse2 = np.sqrt(mean_squared_error(y,y_poly_pred))
print("Root mean square error for Lasso regression from scratch = " + str(rmse2))

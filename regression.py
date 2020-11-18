import requests
import pandas
import scipy
import numpy
import sys
import csv
from sklearn import linear_model

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


	

def normalise(data):
	max=numpy.amax(data)
	min=numpy.amin(data)
	den=max-min
	for i in range(len(data)):
		data[i]=(data[i]-min)/den
	return data

def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    #with open(response.content()) as csv_file:
    cont= response.content.decode('utf-8')
    csv_reader = csv.reader(cont.splitlines(), delimiter=',')
    line_count = 0
    for row in csv_reader:
    	if line_count == 0:
    		x_train=numpy.array(row[1:]).astype(numpy.float)
    		line_count=1
    	else:
    		y_train=numpy.array(row[1:]).astype(numpy.float)
    x_train= normalise(x_train)
    y_train=normalise(y_train)
    model = linear_model.Ridge(alpha = 200)
    model.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
    return model.predict(numpy.array(area).reshape(-1,1))


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

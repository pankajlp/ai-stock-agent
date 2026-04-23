
# models/lstm/utils.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(dataset, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x, y = [], []
    for i in range(time_step, len(scaled_data)):
        x.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y, scaler

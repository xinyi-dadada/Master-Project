import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from cnn_data_prep import CNNDataPrepare

path = 'radar_112/radar_112_all.parquet'
name = 'radar_112'

cnn_prep = CNNDataPrepare(path=path, name=name)
train_set, test_set = cnn_prep.data_prep()


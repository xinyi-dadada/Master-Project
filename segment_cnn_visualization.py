import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocess import DivideData
class SegmentVisual():
    def __init__(self, prediction, data_loader):
        self.output = prediction
        self.data_loader = data_loader

    def Visualization(self):
        for one_batch in self.data_loader:
            signal_data = one_batch[0]
            signal_labels = one_batch[1]
            print('test')



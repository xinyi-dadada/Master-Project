from cnn_data_prep import CNNDataPrepare
from cnn_train import CNNTrain
from cnn_eval import CNNEvaluation
import numpy as np

path = ''
#path = '/home/Shared/xinyi/blob1/thesis/logs_seg/radar112_new_cnn_0710.parquet' #multilabel
name = 'radar_112'

epoch = 50
model_name = f'/home/Shared/xinyi/blob1/thesis/model/model_0710'


cnn_prep = CNNDataPrepare(path=path, name=name)
train_dataloader, test_dataloader = cnn_prep.data_prep()
training = CNNTrain(train_name=name, epoch=epoch, model_name=model_name, fold_path=path, data_loader=train_dataloader)
training.train()
evaluation = CNNEvaluation(torch_model=model_name, epoch=epoch, dataloader=test_dataloader, fold_path=path)
evaluation.CNN_eval()
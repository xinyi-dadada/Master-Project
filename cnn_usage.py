from cnn_data_prep import CNNDataPrepare
from cnn_train import CNNTrain
from cnn_eval import CNNEvaluation
import numpy as np

path = 'radar_112/combined_task_all_0807.parquet'
name = 'radar_112'
batchsize = 4
epoch = 50
model_name = f'model_0807_{name}_batch{batchsize}_with-100000_1'



###############################

cnn_prep = CNNDataPrepare(path=path, name=name, batchsize=batchsize)
train_dataloader, test_dataloader = cnn_prep.data_prep()
training = CNNTrain(train_name=name, epoch=epoch, model_name=model_name, dataloader=train_dataloader)
training.train()
evaluation = CNNEvaluation(torch_model=model_name, epoch=epoch, dataloader=test_dataloader)
evaluation.CNN_eval()


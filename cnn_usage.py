from cnn_data_prep import CNNDataPrepare
from cnn_train import CNNTrain
from cnn_eval import CNNEvaluation

path = ...
name = ...
batchsize = ...
epoch = ...
model_name = ...

###############################
cnn_prep = CNNDataPrepare(path=path, name=name)
train_dataloader, test_dataloader = cnn_prep.data_prep()
training = CNNTrain(train_name=name, epoch=epoch, model_name=model_name, fold_path=path, data_loader=train_dataloader)
training.train()
evaluation = CNNEvaluation(torch_model=model_name, epoch=epoch, dataloader=test_dataloader, fold_path=path)
evaluation.CNN_eval()


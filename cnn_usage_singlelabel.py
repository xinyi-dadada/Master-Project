from cnn_dataprep_singlelabel import CNNDataPrepare
from cnntrain_singlelabel import CNNTrain
from cnneval_singlelabel import CNNEvaluation

path = '/home/Shared/xinyi/blob1/thesis/cnn_114.parquet'
name = 'radar_114'
epoch = 100
model_name = f'model_1711_{name}_epoch{epoch}_radar114'


###############################

cnn_prep = CNNDataPrepare(path=path, name=name)
folds = cnn_prep.data_prep(num_folds=5)
#train_set_loader, test_set_loader = cnn_prep.data_prep()
training = CNNTrain(train_name=name, epoch=epoch, model_name=model_name, dataloader=train_set_loader)
training.train()
evaluation = CNNEvaluation(torch_model=model_name, eval_name=name, epoch=epoch, dataloader=test_set_loader)
evaluation.CNN_eval()

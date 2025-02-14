from cnn_seg import CNNSegment
from segment_cnn_prep import SegmentCNNPrepare
from segment_cnn_train import CNNSegTrain
from segment_cnn_eval import CNNEvaluation
from segment_cnn_visualization import SegmentVisual


data_path_train = '/home/Shared/xinyi/blob1/thesis/radar_112/all_part_seg_train.parquet'
data_path_test = '/home/Shared/xinyi/blob1/thesis/radar_112/all_part_seg_test.parquet'
# task which is significant detected by this radar
important_tasks = ['T_0', 'T_6', 'T_7', 'T_11']

########################################################################

cnn_prep_train = SegmentCNNPrepare(data_path_train, important_tasks)
train_dataloader = cnn_prep_train.prepare_dataloader()

#cnn_prep_test = SegmentCNNPrepare(data_path_test, important_tasks)
#test_dataloader = cnn_prep_test.prepare_dataloader()
model_name = '2509segtrain_6'
epoch = 100
model = CNNSegment()

# train model
# max participant number: 4 -> CUDA out of memory. Tried to allocate 2.04 GiB. GPU

training = CNNSegTrain(epoch=epoch, model_name=model_name, dataloader=train_dataloader, select_num=35)
training.train_model()

# evaluate model
#evaluation = CNNEvaluation(torch_model=model_name, epoch=epoch, dataloader=test_dataloader, select_num=2)
#eval_output = evaluation.CNNSegEval()

# visualize the prediction
#seg_visual = SegmentVisual(prediction=eval_output, data_loader=test_dataloader)
#seg_visual.Visualization()

from cnn_seg import CNNSegment
from segment_cnn_prep import SegmentCNNPrepare
#from torchviz import make_dot
import hiddenlayer as hl
from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter(log_dir=f'./runs')


data_path_test = '/home/Shared/xinyi/blob1/thesis/radar_112/all_part_seg_test.parquet'
important_tasks = ['T_0', 'T_6', 'T_7', 'T_11']
cnn_prep_test = SegmentCNNPrepare(data_path_test, important_tasks)
test_dataloader = cnn_prep_test.prepare_dataloader()
model = CNNSegment()
batch_text, batch_labels = next(iter(test_dataloader))

dummy_input = batch_text
writer.add_graph(model, dummy_input)
writer.close()


"""
yhat = model(batch_text) # Give dummy batch to forward().
#make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")


transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.

graph = hl.build_graph(model, batch_text, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('rnn_hiddenlayer', format='png')

"""
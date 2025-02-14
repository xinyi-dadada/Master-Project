<<<<<<< HEAD
# Master-Project
Project Name: Extracting Activities of Daily Living from Ambient Sensor
=======
# Note

## Data Processing
* [reconstruct_data](data_label_par.py): merge each task per participant together as a nest numpy array into a parquet file
* [preare_data](prepare_data.py): give the radar number, it will find the parquet data and divide by participants and required tasks
* [dataset](dataset.py): merge all the participants and all tasks in one parquet file
* [data](data.py): utils for data processing

## Supervised Learning
* [cnn process data](part_tasks.py): process the raw data per tasks, including 7 channels and task number
* [extract task](task_cnn.py): only extract the task with features
* [cnn usage](cnn_classification_usage.py): run CNN 
* [cnn data prep](cnn_data_prep.py): prepare the dataset for CNN 
* [cnn train](cnn_train.py): training the model
* [cnn evaluate](cnn_eval.py): evaluate the model
* [cnn strcuture](cnn.py): the structure for the neural network

>>>>>>> 923350f (Initial commit)

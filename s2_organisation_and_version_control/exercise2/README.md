Running the system: 

As of now both the train_model.py and predict_model.py runs on PyTorch's own downloaded version of the data
e.g.  test_set = torchvision.datasets.FashionMNIST(...). This was done since a custom dataset / dataloader was 
in development but proved to time consuming, as the core of the exercises was not for this to work. 

This will however be made such that the systems runs on the train_0.npz files later on, this is now an honor thing.

Running the train_model.py

Running this script is straight forward from the terminal

example "python train_model.py train" will train on the model specified in model.py
and will save the trained model in the models "dtu_mlops\s2_organisation_and_version_control\exercise2\models" folder.
and a picture of the training loss in report folder "dtu_mlops\s2_organisation_and_version_control\exercise2\reports\figures".



Running predict_model.py

Example "python predict_model.py --model 2022-01-05/model10-47-40-pt"

this will be updated to include testing data folder as input. 
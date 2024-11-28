from helical import HelixmRNAFineTuningModel, HelixmRNAConfig
import pandas as pd
import torch

data_path = '/home/data/mamba2_benchmark_data/Tc-Riboswitches.csv'
data = pd.read_csv(data_path)
train_data = data[data['Split'] == 'train']
eval_data = data[data['Split'] == 'test']

helixr_config = HelixmRNAConfig(batch_size=5, device='cuda')
helixr_fine_tune = HelixmRNAFineTuningModel(helix_mrna_config=helixr_config, fine_tuning_head='regression', output_size=1)

train_dataset = helixr_fine_tune.process_data(train_data['Sequence'].values)
eval_dataset = helixr_fine_tune.process_data(eval_data['Sequence'].values)

train_labels = train_data["Value"].values.astype('float32').reshape(-1, 1)
eval_labels = eval_data["Value"].values.astype('float32').reshape(-1, 1)

helixr_fine_tune.train(train_dataset=train_dataset, train_labels=train_labels, loss_function=torch.nn.MSELoss(), validation_dataset=eval_dataset, validation_labels=eval_labels)

outputs = helixr_fine_tune.get_outputs(eval_dataset)

print(outputs[:15])
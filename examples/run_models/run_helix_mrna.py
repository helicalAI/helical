from helical import HelixmRNA, HelixmRNAConfig
import pandas as pd


data_path = '/home/data/mamba2_benchmark_data/mRFP_Expression.csv'
data = pd.read_csv(data_path)
train_data = data[data['Split'] == 'train'][:10]

helixr = HelixmRNA(HelixmRNAConfig(batch_size=2, device='cuda'))

dataset = helixr.process_data(train_data['Sequence'])

embeddings = helixr.get_embeddings(dataset)
print(embeddings[:15])
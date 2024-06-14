from helical.models.helical import HelicalBaseModel
from anndata import AnnData
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class Benchmark():
    def __init__(self, models: list[HelicalBaseModel], data: AnnData) -> None:
        self.data = data
        for model in models:
            print(model.__class__.__name__)
            dataset = model.process_data(data)
            self.embeddings = {model.__class__.__name__: model.get_embeddings(dataset) }
        
    def get_all_embeddings(self):
        return self.embeddings
    
    def classification(self):
        y = np.array(self.data.obs["celltype"].tolist())
        num_types = self.data.obs["celltype"].unique().shape[0]

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_encoded = to_categorical(y_encoded, num_classes=num_types)
        y_encoded.shape
        pass
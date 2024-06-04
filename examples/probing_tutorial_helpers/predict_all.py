from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import datetime
from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig    
import json

configurer = HyenaDNAConfig(model_name="hyenadna-tiny-1k-seqlen-d256")

from datasets import get_dataset_config_names
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import sklearn

configs = get_dataset_config_names("InstaDeepAI/nucleotide_transformer_downstream_tasks")
result = []
for i, label in enumerate(configs):
    print("Processing:", label, i+1, "/", len(configs))
    dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", label)
    num_classes = len(set(dataset["train"]["label"]))


    x = np.load(f"data/train/x_{label}_norm_256.npy")
    y = np.load(f"data/train/y_{label}_norm_256.npy")
    # # One-hot encode the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_encoded = to_categorical(y_encoded, num_classes=num_classes)

    # X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0, random_state=42)

    input_shape = (configurer.config['d_model'],)

    # define the model
    head_model = Sequential()
    head_model.add(Dense(256, activation='relu', input_shape=input_shape))
    head_model.add(Dropout(0.4)) 
    head_model.add(Dense(128, activation='relu'))
    head_model.add(Dropout(0.4))
    head_model.add(Dense(128, activation='relu'))
    head_model.add(Dropout(0.4))  
    head_model.add(Dense(64, activation='relu'))
    head_model.add(Dropout(0.4))  
    head_model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    optimizer = Adam(learning_rate=0.001)
    f1_score = F1Score(num_classes, average='macro')

    head_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=f1_score)

    # Setup callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = head_model.fit(x, y_encoded, epochs=40, batch_size=64, callbacks=[tensorboard_callback])

    X_unseen = np.load(f"data/test/x_{label}_norm_256.npy")
    y_unseen = np.load(f"data/test/y_{label}_norm_256.npy")

    predictions_nn = head_model.predict(X_unseen)
    y_pred = np.argmax(predictions_nn, axis=1)

    # accuracy.append(sum(np.equal(y_pred, y_unseen))/len(y_unseen))
    # Matthews correlation coefficient (MCC) for enhancer and epigenetic datasets
    # F1-score for promoter and splice site datasets
    if label in ["H4ac", 
                 "H3K36me3", 
                 "H3", 
                 "H4", 
                 "H3K4me3", 
                 "H3K4me1", 
                 "H3K14ac", 
                 "enhancers_types", 
                 "H3K79me3", 
                 "H3K4me2", 
                 "enhancers", 
                 "H3K9ac"]:
        result.append(matthews_corrcoef(y_unseen, y_pred))
        
    else:
        if num_classes == 2:
            result.append(sklearn.metrics.f1_score(y_unseen, y_pred, average='binary'))
        else:
            # I'm not 100% sure what the average parameter should be here. I dont think the paper specifies it
            # splice_sites_all has 3 classes
            result.append(sklearn.metrics.f1_score(y_unseen, y_pred, average='weighted'))


with open("file.json", 'w') as f:
    json.dump(result, f, indent=2) 
    json.dump(configs, f, indent=2) 





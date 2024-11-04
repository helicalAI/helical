**Example Usage**
```python
from helical import HelixR, HelixRConfig

helixr = HelixR(HelixRConfig(batch_size=5, device='cuda'))

dataset = helixr.process_data(rna_sequence_strings_list)

embeddings = helixr.get_embeddings(dataset)
print("HelixR embeddings shape: ", embeddings.shape)
```

## How To Fine-Tune

```python
from helical import HelixRFineTuningModel, HelixRConfig

helixr_config = HelixRConfig(batch_size=5, device='cuda')
helixr_fine_tune = HelixRFineTuningModel(helixr_config=helixr_config, fine_tuning_head='regression', output_size=1)

train_dataset = helixr_fine_tune.process_data(rna_sequence_strings_list)

helixr_fine_tune.train(train_dataset=train_dataset, train_labels=output_labels)

outputs = helixr_fine_tune.get_outputs(eval_dataset)

print("HelixR fine-tuned model output shape", outputs.shape)
```


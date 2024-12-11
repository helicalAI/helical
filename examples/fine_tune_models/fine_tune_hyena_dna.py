from helical import HyenaDNAFineTuningModel, HyenaDNAConfig
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../run_models/configs", config_name="hyena_dna_config")
def run_fine_tuning(cfg: DictConfig):
    input_sequences = ["ACT"*20, "ATG"*20, "ATG"*20, "ACT"*20, "ATT"*20]
    labels = [0, 2, 2, 0, 1]

    hyena_dna_config = HyenaDNAConfig(**cfg)
    hyena_dna_fine_tune = HyenaDNAFineTuningModel(hyena_config=hyena_dna_config, fine_tuning_head="classification", output_size=3)

    train_dataset = hyena_dna_fine_tune.process_data(input_sequences)

    hyena_dna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = hyena_dna_fine_tune.get_outputs(train_dataset)
    
    print(outputs.shape)

if __name__ == "__main__":
    run_fine_tuning()

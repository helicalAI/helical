from helical.models.helix_mrna import HelixmRNAFineTuningModel, HelixmRNAConfig
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../run_models/configs",
    config_name="helix_mrna_config",
)
def run_fine_tuning(cfg: DictConfig):
    input_sequences = ["EACU" * 20, "EAUG" * 20, "EAUG" * 20, "EACU" * 20, "EAUU" * 20]
    labels = [0, 2, 2, 0, 1]

    helix_mrna_config = HelixmRNAConfig(**cfg)
    helix_mrna_fine_tune = HelixmRNAFineTuningModel(
        helix_mrna_config=helix_mrna_config,
        fine_tuning_head="classification",
        output_size=3,
    )

    train_dataset = helix_mrna_fine_tune.process_data(input_sequences)

    helix_mrna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = helix_mrna_fine_tune.get_outputs(train_dataset)
    print(outputs)

    # save and load model
    helix_mrna_fine_tune.save_model("./helix_mrna_fine_tuned_model.pt")
    helix_mrna_fine_tune.load_model("./helix_mrna_fine_tuned_model.pt")

    outputs = helix_mrna_fine_tune.get_outputs(train_dataset)
    print(outputs)

if __name__ == "__main__":
    run_fine_tuning()

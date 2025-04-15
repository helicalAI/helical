from helical.models.mamba2_mrna import Mamba2mRNAFineTuningModel, Mamba2mRNAConfig
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../run_models/configs",
    config_name="mamba2_mrna_config",
)
def run_fine_tuning(cfg: DictConfig):
    input_sequences = ["ACUG" * 20, "AUGC" * 20, "AUGC" * 20, "ACUG" * 20, "AUUG" * 20]
    labels = [0, 2, 2, 0, 1]

    mamba2_mrna_config = Mamba2mRNAConfig(**cfg)
    mamba2_mrna_fine_tune = Mamba2mRNAFineTuningModel(
        mamba2_mrna_config=mamba2_mrna_config,
        fine_tuning_head="classification",
        output_size=3,
    )

    train_dataset = mamba2_mrna_fine_tune.process_data(input_sequences)

    mamba2_mrna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = mamba2_mrna_fine_tune.get_outputs(train_dataset)
    print(outputs)

    # save and load model
    mamba2_mrna_fine_tune.save_model("./mamba2_mrna_fine_tuned_model.pt")
    mamba2_mrna_fine_tune.load_model("./mamba2_mrna_fine_tuned_model.pt")

    outputs = mamba2_mrna_fine_tune.get_outputs(train_dataset)
    print(outputs)

if __name__ == "__main__":
    run_fine_tuning()

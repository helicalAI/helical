from helical.models.caduceus import CaduceusFineTuningModel, CaduceusConfig
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../run_models/configs",
    config_name="caduceus_config",
)
def run_fine_tuning(cfg: DictConfig):
    input_sequences = ["ACT" * 20, "ATG" * 10, "ATG" * 20, "CTG" * 10, "TTG" * 20]
    labels = [0, 2, 2, 0, 1]

    caduceus_config = CaduceusConfig(**cfg)
    caduceus_fine_tune = CaduceusFineTuningModel(
        caduceus_config=caduceus_config,
        fine_tuning_head="classification",
        output_size=3,
    )

    train_dataset = caduceus_fine_tune.process_data(input_sequences)

    caduceus_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = caduceus_fine_tune.get_outputs(train_dataset)
    print(outputs)

    # save and load model
    caduceus_fine_tune.save_model("./caduceus_fine_tuned_model.pt")
    caduceus_fine_tune.load_model("./caduceus_fine_tuned_model.pt")

    outputs = caduceus_fine_tune.get_outputs(train_dataset)
    print(outputs)  

if __name__ == "__main__":
    run_fine_tuning()

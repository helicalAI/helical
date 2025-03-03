from helical.models.caduceus import Caduceus, CaduceusConfig
import hydra
from omegaconf import DictConfig
from pandas import DataFrame


@hydra.main(version_base=None, config_path="configs", config_name="caduceus_config")
def run(cfg: DictConfig):
    caduceus_config = CaduceusConfig(**cfg)
    caduceus = Caduceus(configurer=caduceus_config)

    data = DataFrame({"Sequence": ["ACTG" * int(1024 / 4), "TGCA" * int(1024 / 4)]})
    processed_data = caduceus.process_data(data)

    embeddings = caduceus.get_embeddings(processed_data)
    print(embeddings.shape)


if __name__ == "__main__":
    run()

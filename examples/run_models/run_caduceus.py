from helical import Caduceus, CaduceusConfig
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="caduceus_config")
def run(cfg: DictConfig):
    caduceus_config = CaduceusConfig(**cfg)
    caduceus = Caduceus(configurer = caduceus_config)
    
    sequence = ['ACTG' * int(1024/4), 'TGCA' * int(1024/4)]
    processed_data = caduceus.process_data(sequence)

    embeddings = caduceus.get_embeddings(processed_data)
    print(embeddings.shape)

if __name__ == "__main__":
    run()
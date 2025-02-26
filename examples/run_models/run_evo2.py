from helical.models.evo_2 import Evo2, Evo2Config
import hydra
from omegaconf import DictConfig
from pandas import DataFrame


@hydra.main(version_base=None, config_path="configs", config_name="evo_2_config")
def run(cfg: DictConfig):
    # Load the Evo2 model
    evo2_config = Evo2Config(**cfg)
    evo2 = Evo2(configurer=evo2_config)

    data = DataFrame({"Sequence": ["ACTG" * int(1024 / 4), "TGCA" * int(1024 / 2)]})

    dataset = evo2.process_data(data)

    embeddings = evo2.get_embeddings(dataset)

    generate = evo2.generate(dataset)
    # Get the last embedding of each sequence
    print(embeddings["embeddings"][0][embeddings["original_lengths"][0]-1])
    print(embeddings["embeddings"][1][embeddings["original_lengths"][1]-1])
    print(embeddings["original_lengths"])

    # Print the generated sequences
    print(generate)

if __name__ == "__main__":
    run()
from helical.models.evo_2 import Evo2, Evo2Config

evo2_config = Evo2Config(batch_size=1, model_name="evo2-1b-base")

evo2 = Evo2(configurer=evo2_config)

print(evo2.model)

sequences = ["ACGT" * 1000]

dataset = evo2.process_data(sequences)

embeddings = evo2.get_embeddings(dataset)

# generate = evo2.generate(sequences)

print(embeddings)

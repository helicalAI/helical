from helical.models.hyena_dna.model import HyenaDNA, HyenaDNAConfig

hyena_config = HyenaDNAConfig(model_name = "hyenadna-tiny-1k-seqlen-d256")
model = HyenaDNA(configurer = hyena_config)   
sequence = 'ACTG' * int(1024/4)
tokenized_sequence = model.process_data(sequence)
embeddings = model.get_embeddings(tokenized_sequence)
print(embeddings.shape)

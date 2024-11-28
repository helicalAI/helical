from helical import Caduceus, CaduceusConfig

caduceus = CaduceusConfig()
model = Caduceus(configurer = caduceus)   
sequence = 'ACTG' * int(1024/4)
# tokenized_sequence = model.process_data(sequence)
# embeddings = model.get_embeddings(tokenized_sequence)
# print(embeddings.shape)
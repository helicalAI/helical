from helical.models.hyena_dna.model import HyenaDNA,HyenaDNAConfig
config = HyenaDNAConfig(model_name="hyenadna-tiny-1k-seqlen-d256")
model = HyenaDNA(model_config=config)   
print("Done")

import hydra
from omegaconf import DictConfig
import anndata as ad
from helical.models.c2s import Cell2Sen, Cell2SenConfig

@hydra.main(version_base=None, config_path="configs", config_name="c2s_config")
def run(cfg: DictConfig):

    adata = ad.read_h5ad("./yolksac_human.h5ad")
    n_cells = 1
    n_genes = 200
    adata = adata[:n_cells, :n_genes].copy()
    perturbation_column = "perturbation"
    adata.obs[perturbation_column] = ["IFNg"] * n_cells

    config = Cell2SenConfig(**cfg)
    c2s = Cell2Sen(configurer=config)

    processed_dataset = c2s.process_data(adata)
    embeddings, attentions, genes_names_attn = c2s.get_embeddings(processed_dataset,output_attentions=True)
    perturbed_dataset, perturbed_cell_sentences = c2s.get_perturbations(processed_dataset)
    # Print the first cell sentence and its words for comparison
    first_sentence = processed_dataset['cell_sentence'][0]
    words = first_sentence.split()
    print(f"\nFirst cell sentence ({len(words)} words): {first_sentence}")

    # Show how each gene gets tokenized into subtokens
    print("\nGene -> subtokens:")
    for gene in words:
        token_ids = c2s.tokenizer.encode(gene, add_special_tokens=False)
        subtokens = c2s.tokenizer.convert_ids_to_tokens(token_ids)
        print(f"  {gene:>20s} -> {subtokens}")
    print(f"  Total genes: {len(words)}, Total subtokens (genes only): {sum(len(c2s.tokenizer.encode(g, add_special_tokens=False)) for g in words)}")

    # attentions is a tuple of lists: attentions[layer][sample] -> (num_heads, num_words, num_words)
    print(f"\nNumber of layers: {len(attentions)}")
    print(f"Number of samples: {len(attentions[0])}")
    print(f"First sample, first layer shape: {attentions[0][0].shape}")

if __name__ == "__main__":
    run()

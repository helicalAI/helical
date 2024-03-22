from gene_embeddings import load_gene_embeddings_adata
import scanpy as sc
from torch.utils.data import Dataset,DataLoader
from typing import Dict
import torch
import numpy as np
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from uce_model import TransformerModel
from torch import nn
import scipy

class UCECollator(object):
    def __init__(self, config):
        self.pad_length = config['pad_length']


    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))

        idxs = torch.zeros(batch_size)

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx

            i += 1

        return batch_sentences[:, :max_len] , mask[:, :max_len], idxs, cell_sentences
    
class UCEDataset(Dataset):
    def __init__(self, sorted_dataset_names,
                 shapes_dict,
                 model_config, 
                 dataset_to_protein_embeddings,
                 datasets_to_chroms,
                 datasets_to_starts,
                 expression_counts_path) -> None:
        super(UCEDataset, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.config = model_config
        self.collator_fn = UCECollator(model_config)

        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            # self.xs[name] = X
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets = sorted_dataset_names

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = dataset_to_protein_embeddings
        self.dataset_to_chroms = datasets_to_chroms
        self.dataset_to_starts = datasets_to_starts
        self.npzs_dir = expression_counts_path

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    cts = np.memmap(self.npzs_dir + f"{dataset}_counts.npz", dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    counts = cts[idx]
                    counts = torch.tensor(counts).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    batch_sentences, mask, seq_len, cell_sentences = \
                        self.sample_cell_sentences(counts, weights)
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes
    
    def sample_cell_sentences(self,counts, batch_weights):

        dataset_idxs = self.dataset_to_protein_embeddings # get the dataset specific protein embedding idxs
        cell_sentences = torch.zeros((counts.shape[0], self.config["pad_length"])) # init the cell representation as 0s
        mask = torch.zeros((counts.shape[0], self.config["pad_length"])) # start of masking the whole sequence
        chroms = self.dataset_to_chroms # get the dataset specific chroms for each gene
        starts = self.dataset_to_starts # get the dataset specific genomic start locations for each gene

        longest_seq_len = 0 # we need to keep track of this so we can subset the batch at the end
        for c, cell in enumerate(counts):
            weights = batch_weights[c].numpy()
            weights = weights / sum(weights)  # RE NORM after mask
            
            # randomly choose the genes that will make up the sample, weighted by expression, with replacement
            choice_idx = np.random.choice(np.arange(len(weights)),
                                        size=self.config["sample_size"], p=weights,
                                        replace=True)

            choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
            # order the genes by chromosome
            chrom_sort = np.argsort(choosen_chrom)  
            choice_idx = choice_idx[chrom_sort]

            # sort the genes by start
            new_chrom = chroms[choice_idx]
            choosen_starts = starts[choice_idx]

            ordered_choice_idx = np.full((self.config["pad_length"]),
                                        self.config["cls_token_idx"])  # start with cls
            # i= 0 first token is CLS
            i = 1  # continue on to the rest of the sequence with left bracket being assumed.
            # Shuffle the chroms now, there's no natural order to chromosomes
            uq_chroms = np.unique(new_chrom)
            np.random.shuffle(uq_chroms) # shuffle
            
            # This loop is actually just over one cell
            for chrom in uq_chroms:
                # Open Chrom token
                ordered_choice_idx[i] = int(chrom) + self.config["CHROM_TOKEN_OFFSET"] # token of this chromosome # i = 1 next token is a chrom open
                i += 1
                # now sort the genes by start order within the chroms
                loc = np.where(new_chrom == chrom)[0]
                sort_by_start = np.argsort(
                    choosen_starts[loc])  # start locations for this chromsome

                to_add = choice_idx[loc[sort_by_start]]
                ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
                i += len(to_add)
                ordered_choice_idx[i] = self.config["chrom_token_right_idx"] # add the chrom sep again
                i += 1  # add the closing token again

            longest_seq_len = max(longest_seq_len, i)
            remainder_len = (self.config["pad_length"] - i)

            cell_mask = torch.concat((torch.ones(i),
                                    # pay attention to all of these tokens, ignore the rest!
                                    torch.zeros(remainder_len)))

            mask[c, :] = cell_mask

            ordered_choice_idx[i:] = self.config["pad_token_idx"] # the remainder of the sequence
            cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
            
        cell_sentences_pe = cell_sentences.long() # token indices
        
        return cell_sentences_pe, mask, longest_seq_len, cell_sentences

def process_data(anndata,model_config,species='human',filter_genes=False,accelerator=None):
        if filter_genes:
            sc.pp.filter_genes(anndata, min_cells=10)
            # sc.pp.filter_cells(ad, min_genes=25)
        ##Filtering out the Expression Data That we do not have in the protein embeddings
        adata, protein_embeddings,full_pe_dict = load_gene_embeddings_adata(
        adata=anndata,
        species=[species],
        embedding_model="ESM2")
        protein_embeddings = protein_embeddings[species]
        if scipy.sparse.issparse(adata.X):
            expression = np.asarray(adata.X.todense())
        else:
            expression = np.asarray(adata.X)


        # print(expression.max()) # this is a nice check to make sure it's counts
        npz_folder_path = "./"
        name = "test"
        filename = npz_folder_path + f"{name}_counts.npz"
        shape = expression.shape
        fp = np.memmap(filename, dtype='int64', mode='w+', shape=shape)
        fp[:] = expression[:]
        fp.flush()
        
        num_cells = adata.X.shape[0]
        num_genes = adata.X.shape[1]

        shapes_dict = {name: (num_cells, num_genes)}
        dataset_species = species
        spec_pe_genes = list(full_pe_dict[dataset_species].keys())
        chromosome_list =  pd.read_csv("./model_files/species_chrom.csv")
        chromosome_list["spec_chrom"] = pd.Categorical(chromosome_list["species"] + "_" +  chromosome_list["chromosome"]) # add the spec_chrom list
        with open("./model_files/species_offsets.pkl", "rb") as f:
            species_to_offsets = pickle.load(f)
        offset = species_to_offsets[dataset_species]
        pe_row_idxs = torch.tensor([spec_pe_genes.index(k.lower()) + offset for k in adata.var_names]).long()
        # print(len(np.unique(pe_row_idxs)))
        
        spec_chrom = chromosome_list[chromosome_list["species"] == dataset_species].set_index("gene_symbol")

        gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

        dataset_chroms = gene_chrom["spec_chrom"].cat.codes # now this is correctely indexed by species and chromosome
        # print("Max Code:", max(dataset_chroms))
        dataset_pos = gene_chrom["start"].values
    
        dataset = UCEDataset(sorted_dataset_names=['test'],
                                shapes_dict=shapes_dict,
                                model_config=model_config, 
                                expression_counts_path="./",
                                dataset_to_protein_embeddings=pe_row_idxs,
                                datasets_to_chroms=dataset_chroms,
                                datasets_to_starts=dataset_pos
                                )

        dataloader = DataLoader(dataset, batch_size=5, shuffle=False,
                                collate_fn=dataset.collator_fn,
                                num_workers=0)
        
        if accelerator is not None:
            dataloader = accelerator.prepare(dataloader)
        #     # pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        # else:
        #     # pbar = tqdm(dataloader)
        return dataloader



def get_ESM2_embeddings(files):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(files['token_file'])
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, files['token_dim']))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack(
            (all_pe, CHROM_TENSORS))  # Add the chrom tensors to the end
        all_pe.requires_grad = False

    return all_pe



## writing a funciton to load the model 
def load_model(model_config,all_pe):
    token_dim = model_config["token_dim"]
    emsize = 1280  # embedding dimension
    d_hid = model_config['d_hid']  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = model_config['n_layers']
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    output_dim = model_config['output_dim']
    model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                            d_hid=d_hid,
                            nlayers=nlayers, dropout=dropout,
                            output_dim=output_dim)

    # empty_pe = torch.zeros(50000, 5120)
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(model_config['model_loc'], map_location="cpu"),
                            strict=True)
    if all_pe.shape[0] != 145469: 
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    
    return model


# Create a function that uses the model to get the embeddings of the genes
def get_gene_embeddings(model, dataloader, accelerator,model_config=None):
    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    dataset_embeds = []
    with torch.no_grad():
        for batch in pbar:
            batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
            batch_sentences = batch_sentences.permute(1, 0)
            if model_config and model_config["multi_gpu"]:
                batch_sentences = model.module.pe_embedding(batch_sentences.long())
            else:
                batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences,
                                                      dim=2)  # Normalize token outputs now
            _, embedding = model.forward(batch_sentences, mask=mask)
            # Fix for duplicates in last batch
            accelerator.wait_for_everyone()
            embeddings = accelerator.gather_for_metrics((embedding))
            if accelerator.is_main_process:
                dataset_embeds.append(embeddings.detach().cpu().numpy())

            ##Onlye 1 batch
            # break
    return np.vstack(dataset_embeds)
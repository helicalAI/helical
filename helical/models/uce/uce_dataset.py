from helical.models.uce.uce_collator import UCECollator
from torch.utils.data import Dataset
import numpy as np
import torch
from typing import Dict

class UCEDataset(Dataset):
    def __init__(self, 
                 sorted_dataset_names,
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
            
            # set the seed for reproducibility
            generator = np.random.default_rng(seed=42)

            # randomly choose the genes that will make up the sample, weighted by expression, with replacement
            choice_idx = generator.choice(np.arange(len(weights)),
                                        size=self.config["sample_size"], p=weights,
                                        replace=True)

            choosen_chrom = chroms.iloc[choice_idx] # get the sampled genes chromosomes
            # order the genes by chromosome
            chrom_sort = np.argsort(choosen_chrom)  
            choice_idx = choice_idx[chrom_sort]

            # sort the genes by start
            new_chrom = chroms.iloc[choice_idx]
            choosen_starts = starts[choice_idx]

            ordered_choice_idx = np.full((self.config["pad_length"]),
                                        self.config["cls_token_idx"])  # start with cls
            # i= 0 first token is CLS
            i = 1  # continue on to the rest of the sequence with left bracket being assumed.
            # Shuffle the chroms now, there's no natural order to chromosomes
            uq_chroms = np.unique(new_chrom)
            generator.shuffle(uq_chroms) # shuffle
            
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

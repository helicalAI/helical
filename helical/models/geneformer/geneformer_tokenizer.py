"""
Geneformer tokenizer.

**Input data:**

| *Required format:* raw counts scRNAseq data without feature selection as .loom or anndata file.
| *Required row (gene) attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required col (cell) attribute:* "total_counts"; total read counts in that cell.

| *Optional col (cell) attribute:* "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

**Usage:**

.. code-block :: python

    >>> from geneformer import TranscriptomeTokenizer
    >>> tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ"}, nproc=4)
    >>> tk.tokenize_data("data_directory", "output_directory", "output_prefix")

**Description:**

| Input data is a directory with .loom or .h5ad files containing raw counts from single cell RNAseq data, including all genes detected in the transcriptome without feature selection. 
  The input file type is specified by the argument file_format in the tokenize_data function.

| The discussion below references the .loom file format, but the analagous labels are required for .h5ad files, just that they will be column instead of row attributes and vice versa due to the transposed format of the two file types.

| Genes should be labeled with Ensembl IDs (loom row attribute "ensembl_id"), which provide a unique identifer for conversion to tokens. Other forms of gene annotations (e.g. gene names) can be converted to Ensembl IDs via Ensembl Biomart. 
  Cells should be labeled with the total read count in the cell (loom column attribute "total_counts") to be used for normalization.

| No cell metadata is required, but custom cell attributes may be passed onto the tokenized dataset by providing a dictionary of custom attributes to be added, which is formatted as loom_col_attr_name : desired_dataset_col_attr_name. 
  For example, if the original .loom dataset has column attributes "cell_type" and "organ_major" and one would like to retain these attributes as labels in the tokenized dataset with the new names "cell_type" and "organ", respectively, 
  the following custom attribute dictionary should be provided: {"cell_type": "cell_type", "organ_major": "organ"}.

| Additionally, if the original .loom file contains a cell column attribute called "filter_pass", this column will be used as a binary indicator of whether to include these cells in the tokenized data. 
  All cells with "1" in this attribute will be tokenized, whereas the others will be excluded. One may use this column to indicate QC filtering or other criteria for selection for inclusion in the final tokenized dataset.

| If one's data is in other formats besides .loom or .h5ad, one can use the relevant tools (such as Anndata tools) to convert the file to a .loom or .h5ad format prior to running the transcriptome tokenizer.

| OF NOTE: Take care that the correct token dictionary and gene median file is used for the correct model. 

| OF NOTE: For 95M model series, special_token should be True and model_input_size should be 4096. For 30M model series, special_token should be False and model_input_size should be 2048.

"""

from __future__ import annotations

import os
import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal
from collections import Counter

import loompy as lp
import numpy as np
import scipy.sparse as sp
from datasets import Dataset
from anndata import AnnData
from numpy import ndarray
import pandas as pd
from tqdm import tqdm
import scanpy as sc

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
LOGGER = logging.getLogger(__name__)


def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


def sum_ensembl_ids(
    data_directory,
    collapse_gene_ids,
    gene_mapping_dict,
    gene_token_dict,
    file_format="loom",
    chunk_size=512,
):
    """
    Process genomic data by mapping and potentially collapsing Ensembl IDs.

    This function handles both Loom and H5AD file formats. It maps Ensembl IDs
    using a provided dictionary and, if necessary, collapses and sums counts for
    duplicate IDs. If no duplicate IDs are found and collapsing is not necessary, the function
      returns the original data without modifications.
    """
    
    if file_format == "loom":
        """
        Map Ensembl IDs from gene mapping dictionary. If duplicate Ensembl IDs are found, sum counts together.
        """
        with lp.connect(data_directory) as data:
            assert (
                "ensembl_id" in data.ra.keys()
            ), "'ensembl_id' column missing from data.ra.keys()"
            gene_ids_in_dict = [
                gene for gene in data.ra.ensembl_id if gene in gene_token_dict.keys()
            ]
            if len(gene_ids_in_dict) == len(set(gene_ids_in_dict)):
                token_genes_unique = True
            else:
                token_genes_unique = False
            if collapse_gene_ids is False:
                if token_genes_unique:
                    return data_directory
                else:
                    raise ValueError("Error: data Ensembl IDs non-unique.")

            gene_ids_collapsed = [
                gene_mapping_dict.get(str(gene_id).upper()) for gene_id in data.ra.ensembl_id
            ]
            gene_ids_collapsed_in_dict = [
                gene for gene in gene_ids_collapsed if gene in gene_token_dict.keys()
            ]

            if (
                len(set(gene_ids_collapsed_in_dict)) == len(set(gene_ids_in_dict))
            ) and token_genes_unique:
                return data_directory
            else:
                dedup_filename = data_directory.with_name(
                    data_directory.stem + "__dedup.loom"
                )
                data.ra["gene_ids_collapsed"] = gene_ids_collapsed
                dup_genes = [
                    idx
                    for idx, count in Counter(data.ra["gene_ids_collapsed"]).items()
                    if count > 1
                ]
                num_chunks = int(np.ceil(data.shape[1] / chunk_size))
                first_chunk = True
                for _, _, view in tqdm(
                    data.scan(axis=1, batch_size=chunk_size), total=num_chunks
                ):

                    def process_chunk(view, duplic_genes):
                        data_count_view = pd.DataFrame(
                            view, index=data.ra["gene_ids_collapsed"]
                        )
                        unique_data_df = data_count_view.loc[
                            ~data_count_view.index.isin(duplic_genes)
                        ]
                        dup_data_df = data_count_view.loc[
                            data_count_view.index.isin(
                                [i for i in duplic_genes if "None" not in i]
                            )
                        ]
                        summed_data = dup_data_df.groupby(dup_data_df.index).sum()
                        if not summed_data.index.is_unique:
                            raise ValueError(
                                "Error: Ensembl IDs in summed data frame non-unique."
                            )
                        data_count_view = pd.concat(
                            [unique_data_df, summed_data], axis=0
                        )
                        if not data_count_view.index.is_unique:
                            raise ValueError(
                                "Error: Ensembl IDs in final data frame non-unique."
                            )
                        return data_count_view

                    processed_chunk = process_chunk(view[:, :], dup_genes)
                    processed_array = processed_chunk.to_numpy()
                    new_row_attrs = {"ensembl_id": processed_chunk.index.to_numpy()}

                    if "n_counts" not in view.ca.keys():
                        total_count_view = np.sum(view[:, :], axis=0).astype(int)
                        view.ca["n_counts"] = total_count_view

                    if first_chunk:  # Create the Loom file with the first chunk
                        lp.create(
                            f"{dedup_filename}",
                            processed_array,
                            row_attrs=new_row_attrs,
                            col_attrs=view.ca,
                        )
                        first_chunk = False
                    else:  # Append subsequent chunks
                        with lp.connect(dedup_filename, mode="r+") as dsout:
                            dsout.add_columns(processed_array, col_attrs=view.ca)
                return dedup_filename

    elif file_format == "h5ad":
        """
        Map Ensembl IDs from gene mapping dictionary. If duplicate Ensembl IDs are found, sum counts together.
        Returns adata object with deduplicated Ensembl IDs.
        """

        data = sc.read_h5ad(str(data_directory))

        assert (
            "ensembl_id" in data.var.columns
        ), "'ensembl_id' column missing from data.var"
        gene_ids_in_dict = [
            gene for gene in data.var.ensembl_id if gene in gene_token_dict.keys()
        ]
        if len(gene_ids_in_dict) == len(set(gene_ids_in_dict)):
            token_genes_unique = True
        else:
            token_genes_unique = False
        if collapse_gene_ids is False:
            if token_genes_unique:
                return data
            else:
                raise ValueError("Error: data Ensembl IDs non-unique.")

        gene_ids_collapsed = [
            gene_mapping_dict.get(str(gene_id).upper()) for gene_id in data.var.ensembl_id
        ]
        gene_ids_collapsed_in_dict = [
            gene for gene in gene_ids_collapsed if gene in gene_token_dict.keys()
        ]
        if (
            len(set(gene_ids_collapsed_in_dict)) == len(set(gene_ids_in_dict))
        ) and token_genes_unique:
            return data

        else:
            data.var["gene_ids_collapsed"] = gene_ids_collapsed
            data.var_names = gene_ids_collapsed
            data = data[:, ~data.var.index.isna()]
            dup_genes = [
                idx for idx, count in Counter(data.var_names).items() if count > 1
            ]

            num_chunks = int(np.ceil(data.shape[0] / chunk_size))

            processed_genes = []
            for i in tqdm(range(num_chunks)):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, data.shape[0])
                data_chunk = data[start_idx:end_idx, :]

                processed_chunks = []
                for dup_gene in dup_genes:
                    data_dup_gene = data_chunk[:, data_chunk.var_names == dup_gene]
                    df = pd.DataFrame.sparse.from_spmatrix(
                        data_dup_gene.X,
                        index=data_dup_gene.obs_names,
                        columns=data_dup_gene.var_names,
                    )
                    df_sum = pd.DataFrame(df.sum(axis=1))
                    df_sum.columns = [dup_gene]
                    df_sum.index = data_dup_gene.obs.index
                    processed_chunks.append(df_sum)

                processed_chunks = pd.concat(processed_chunks, axis=1)
                processed_genes.append(processed_chunks)
            processed_genes = pd.concat(processed_genes, axis=0)
            var_df = pd.DataFrame({"gene_ids_collapsed": processed_genes.columns})
            var_df.index = processed_genes.columns
            processed_genes = sc.AnnData(X=processed_genes, obs=data.obs, var=var_df)

            data_dedup = data[:, ~data.var.index.isin(dup_genes)]  # Deduplicated data
            data_dedup = sc.concat([data_dedup, processed_genes], axis=1)
            data_dedup.obs = data.obs
            data_dedup.var = data_dedup.var.rename(
                columns={"gene_ids_collapsed": "ensembl_id"}
            )
            return data_dedup


DEFAULT_GENE_MEDIAN_FILE = Path(__file__).parent / "v1" / "gene_median_dictionary.pkl"
DEFAULT_TOKEN_DICTIONARY_FILE = Path(__file__).parent / "v1" / "token_dictionary.pkl"
DEFAULT_ENSEMBL_MAPPING_FILE = Path(__file__).parent / "v1" / "ensembl_mapping_dictionary.pkl"
class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1,
        chunk_size=512,
        model_input_size=2048,
        special_token=False,
        collapse_gene_ids=True,
        gene_median_file=DEFAULT_GENE_MEDIAN_FILE,
        token_dictionary_file=DEFAULT_TOKEN_DICTIONARY_FILE,
        gene_mapping_file=DEFAULT_ENSEMBL_MAPPING_FILE,
    ):
        """
        Initialize tokenizer.
        
        **Parameters:**
        
        custom_attr_name_dict : None, dict
            | Dictionary of custom attributes to be added to the dataset.
            | Keys are the names of the attributes in the loom file.
            | Values are the names of the attributes in the dataset.
        nproc : int
            | Number of processes to use for dataset mapping.
        chunk_size : int = 512
            | Chunk size for anndata tokenizer.
        model_input_size : int = 4096
            | Max input size of model to truncate input to.
            | For the 30M model series, should be 2048. For the 95M model series, should be 4096.
        special_token : bool = True
            | Adds CLS token before and EOS token after rank value encoding.
            | For the 30M model series, should be False. For the 95M model series, should be True.
        collapse_gene_ids : bool = True
            | Whether to collapse gene IDs based on gene mapping dictionary.
        gene_median_file : Path
            | Path to pickle file containing dictionary of non-zero median
            | gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            | Path to pickle file containing token dictionary (Ensembl IDs:token).
        gene_mapping_file : None, Path
            | Path to pickle file containing dictionary for collapsing gene IDs.
        """

        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # chunk size for anndata tokenizer
        self.chunk_size = chunk_size

        # input size for tokenization
        self.model_input_size = model_input_size

        # add CLS and EOS tokens
        self.special_token = special_token

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # check for special token in gene_token_dict
        if self.special_token:
            if ("<cls>" not in self.gene_token_dict.keys()) and (
                "<eos>" not in self.gene_token_dict.keys()
            ):
                LOGGER.error(
                    "<cls> and <eos> required in gene_token_dict when special_token = True."
                )
                raise

        if not self.special_token:
            if ("<cls>" in self.gene_token_dict.keys()) and (
                "<eos>" in self.gene_token_dict.keys()
            ):
                LOGGER.warning(
                    "<cls> and <eos> are in gene_token_dict but special_token = False. Please note that for 95M model series, special_token should be True."
                )

        # if collapsing duplicate gene IDs
        self.collapse_gene_ids = collapse_gene_ids

        # load gene mappings dictionary (Ensembl IDs:Ensembl ID)
        if gene_mapping_file is not None:
            with open(gene_mapping_file, "rb") as f:
                self.gene_mapping_dict = pickle.load(f)
        else:
            self.gene_mapping_dict = {k: k for k, _ in self.gene_token_dict.items()}

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_token_dict.keys())

        #  Filter gene mapping dict for items that exist in gene_token_dict
        gene_keys_set = set(self.gene_token_dict.keys())
        self.gene_mapping_dict = {
            k: v for k, v in self.gene_mapping_dict.items() if v in gene_keys_set
        }

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))


    def tokenize_data(
        self,
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["loom", "h5ad"] = "loom",
        use_generator: bool = False,
    ):
        """
        Tokenize .loom files in data_directory and save as tokenized .dataset in output_directory.
        
        **Parameters:**
        
        data_directory : Path
            | Path to directory containing loom files or anndata files
        output_directory : Path
            | Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            | Prefix for output .dataset
        file_format : str
            | Format of input files. Can be "loom" or "h5ad".
        use_generator : bool
            | Whether to use generator or dict for tokenization.
        """
        tokenized_cells, cell_metadata = self.tokenize_files(
            Path(data_directory), file_format
        )
        tokenized_dataset = self.create_dataset(
            tokenized_cells,
            cell_metadata,
            use_generator=use_generator,
        )

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(str(output_path))


    def tokenize_files(
        self, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.values()
            }

        # loops through directories to tokenize .loom files
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )
        for file_path in data_directory.glob(f"*.{file_format}"):
            file_found = 1
            LOGGER.info(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[
                        k
                    ]
            else:
                cell_metadata = None

        if file_found == 0:
            LOGGER.error(
                f"No .{file_format} files found in directory {data_directory}."
            )
            raise
        return tokenized_cells, cell_metadata

    def tokenize_anndata(self, adata_file_path, target_sum=10_000):
        adata = sum_ensembl_ids(
            adata_file_path,
            self.collapse_gene_ids,
            self.gene_mapping_dict,
            self.gene_token_dict,
            file_format="h5ad",
            chunk_size=self.chunk_size,
        )

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in adata.var["ensembl_id"].iloc[coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = adata.var["ensembl_id"].iloc[coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            _ = adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where([i == 1 for i in adata.obs["filter_pass"]])[0]
        elif not var_exists:
            LOGGER.info(
                f"{adata_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), self.chunk_size):
            idx = filter_pass_loc[i : i + self.chunk_size]

            # Check if 'n_counts' exists in adata.obs
            if 'n_counts' in adata.obs.columns:
                n_counts = adata[idx].obs["n_counts"].values[:, None]
            else:
                # If 'n_counts' doesn't exist, calculate it as the sum of counts for each cell
                n_counts = np.sum(adata[idx, :].X, axis=1)[:, None]

            X_view0 = adata[idx, :].X
            X_view = X_view0[:, coding_miRNA_loc]
            X_norm = X_view / n_counts * target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                for i in range(X_norm.shape[0])
            ]

            # add custom attributes for subview to dict
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata
    

    def tokenize_loom(self, loom_file_path, target_sum=10_000):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        dedup_filename = loom_file_path.with_name(loom_file_path.stem + "__dedup.loom")
        loom_file_path = sum_ensembl_ids(
            loom_file_path,
            self.collapse_gene_ids,
            self.gene_mapping_dict,
            self.gene_token_dict,
            file_format="loom",
            chunk_size=self.chunk_size,
        )

        with lp.connect(str(loom_file_path)) as data:
            # Convert data.ra["ensembl_id"] to a numpy array
            ensembl_ids = np.array(data.ra["ensembl_id"])
            
            # Create a boolean mask for coding_miRNA genes
            coding_miRNA_mask = np.array([self.genelist_dict.get(i, False) for i in ensembl_ids])
            coding_miRNA_loc = np.where(coding_miRNA_mask)[0]
                    
            norm_factor_vector = np.array([
                self.gene_median_dict[i]
                for i in ensembl_ids[coding_miRNA_loc]
            ])
            
            coding_miRNA_ids = ensembl_ids[coding_miRNA_loc]
            coding_miRNA_tokens = np.array([self.gene_token_dict[i] for i in coding_miRNA_ids])

            if coding_miRNA_tokens.size == 1:
                coding_miRNA_tokens = np.array([coding_miRNA_tokens.item()])


            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except KeyError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                filter_pass_loc = np.where([i == 1 for i in data.ca["filter_pass"]])[0]
            elif not var_exists:
                LOGGER.info(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for _ix, _selection, view in data.scan(
                items=filter_pass_loc, axis=1, batch_size=self.chunk_size
            ):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / subview.ca.n_counts
                    * target_sum
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        if str(dedup_filename) == str(loom_file_path):
            os.remove(str(dedup_filename))

        return tokenized_cells, file_cell_metadata

    def create_dataset(
        self,
        tokenized_cells,
        cell_metadata,
        use_generator=False,
        keep_uncropped_input_ids=False,
    ):
        LOGGER.info("Creating dataset.")
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:

            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        def format_cell_features(example):
            # Store original uncropped input_ids in separate feature
            if keep_uncropped_input_ids:
                example["input_ids_uncropped"] = example["input_ids"]
                example["length_uncropped"] = len(example["input_ids"])

            # Truncate/Crop input_ids to input size
            if self.special_token:
                example["input_ids"] = example["input_ids"][
                    0 : self.model_input_size - 2
                ]  # truncate to leave space for CLS and EOS token
                example["input_ids"] = np.insert(
                    example["input_ids"], 0, self.gene_token_dict.get("<cls>")
                )
                example["input_ids"] = np.insert(
                    example["input_ids"],
                    len(example["input_ids"]),
                    self.gene_token_dict.get("<eos>"),
                )
            else:
                # Truncate/Crop input_ids to input size
                example["input_ids"] = example["input_ids"][0 : self.model_input_size]
            example["length"] = len(example["input_ids"])

            return example

        output_dataset_truncated = output_dataset.map(
            format_cell_features, num_proc=self.nproc
        )
        return output_dataset_truncated

from helical.services.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import pickle as pkl
import pandas as pd
import numpy as np
import anndata as ad
import tiledbsoma.io
import tiledbsoma

class Preprocessor(Logger):
    def __init__(self, loging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(loging_type, level)
        self.log = logging.getLogger("Preprocessor")

    def map_ensbl_to_name(self, input_path: str, mapping_path: str, count_column: str) -> pd.DataFrame:
        '''
        Maps the 'Ensemble ID' to a lowerscore 'Name' via the provided mapping.
        Nan if no mapping is found.

        Args:
            input_path: Path to the ETS csv file.
            mapping_path: Path to the mapping pickle file.
            count_column: string. The name of the column with the counts.
                        Examples are raw counts 'rcnt' or 'tpm'.
        
        Returns:
            A pandas dataframe representing the gene expression table.
        '''
        mapping = pkl.load(open(mapping_path, 'rb'))
        input = pd.read_csv(input_path)

        self.log.info(f"Starting to do the mapping.")

        input.dropna(subset=['egid'], inplace=True)
        input['gene_name'] = input['egid'].apply(lambda x: mapping[x].get('display_name',np.nan))
        input['gene_name'] = input['gene_name'].apply(lambda x: x.lower() if type(x) is str else x)
        input.dropna(subset=['gene_name'], inplace=True)
        input.set_index(['subject','sample','gene_name'],inplace=True)

        return input[count_column].reset_index()
    
    def transform_table(self, input_path: str, output_path: str, mapping_path: str, count_column: str):
        '''
        Tiledb SOMA expects columns to be the features. These can represent genes, proteins
        or genomic regions. Rows represent observations, which are typically cells. 
        This function transforms a dataframe with genes and raw counts in rows to the desired
        Tiledb SOMA format.
        TODO: What else does it do?
        
        Args:
            input_path: Path to the ETS csv file.
            output_path: Path to the output h5ad AnnData file. 
            mapping_path: Path to the mapping pickle file.
            count_column: string. The name of the column with the counts.
                        Default is the raw count 'rcnt'.
        '''
        gene_expressions = self.map_ensbl_to_name(input_path, mapping_path, count_column)

        self.log.info(f"Successfully received the expression table.")
        self.log.info(f"Converting the expression table to TileDB Soma format.")

        full_df = pd.DataFrame()
        for i, group in gene_expressions.sort_values(['gene_name']).groupby(['subject','sample']):
            group = group.sort_values(['gene_name',count_column])[[count_column,'gene_name']].drop_duplicates(subset='gene_name',keep='last').dropna().set_index('gene_name').T
            full_df = pd.concat([full_df, group], axis=0)

        adata = ad.AnnData(X = full_df)
        adata.write_h5ad(output_path)
        self.log.info(f"Successfully saved the expression table in AnnData h5ad format: {output_path}.")

    def generate_tiledb_soma(self, input_path: str, tiledb_folder_name: str, measurement_name: str):
        '''
        Generates a tiledb soma database

        Args:
            input_path: Path to the h5ad file.
            tiledb_folder_name: The name of the folder where the database will be.
            measurement_name: The name of the measurement.
        '''
        tiledbsoma.io.from_h5ad(experiment_uri = tiledb_folder_name, input_path = input_path, measurement_name = measurement_name)
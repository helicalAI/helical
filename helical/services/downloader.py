import requests
import json
import pickle as pkl
import pandas as pd
from helical.services.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import sys
from pathlib import Path
from git import Repo

INTERVAL = 1000 # interval to get gene mappings
CHUNK_SIZE = 8192 # size of individual chunks to download
LOADING_BAR_LENGTH = 50 # size of the download progression bar in console

class Downloader(Logger):
    def __init__(self, loging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(loging_type, level)
        self.log = logging.getLogger("Downloader")

    def get_ensemble_mapping(self, path_to_ets_csv: Path, output: Path):
        '''
        Saves a mapping of the `Ensemble ID` to `display names`. 
        
        Args:
            path_to_ets_csv: Path to the ETS csv file.
            output: Path to where the output (.pkl) file should be saved to.
        '''
        try:
            df = pd.read_csv(path_to_ets_csv)
        except:
            self.log.exception(f"Failed to open the '{path_to_ets_csv}' file. Please provide it.")

        if output.is_file():
            self.log.info(f"No mapping is done because mapping file already exists here: '{output}'")

        else:
            genes = df['egid'].dropna().unique()

            server = "https://rest.ensembl.org/lookup/id"
            headers={ "Content-Type" : "application/json", "Accept" : "application/json"}

            ensemble_to_display_name = dict()
            
            self.log.info(f"Starting to download the mappings of {len(genes)} genes from '{server}'")

            # Resetting for visualization
            self.data_length = 0
            self.total_length = len(genes)

            for i in range(0, len(genes), INTERVAL):
                self._display_download_progress(INTERVAL)
                ids = {'ids':genes[i:i+INTERVAL].tolist()}
                r = requests.post(server, headers=headers, data=json.dumps(ids))
                decoded = r.json()
                ensemble_to_display_name.update(decoded)

            pkl.dump(ensemble_to_display_name, open(output, 'wb')) 
            self.log.info(f"Downloaded all mappings and saved to: '{output}'")

    def download_via_link(self, output: Path, link: str) -> None:
        '''
        Download a file via a link. 
        
        Args:
            output: Path to the output file.
            link: URL to download the file from.
        '''
       
        if output.is_file():
            self.log.info(f"File: '{output}' exists already. File is not overwritten and nothing is downloaded.")

        else:
            self.log.info(f"Starting to download: '{link}'")
            with open(output, "wb") as f:
                response = requests.get(link, stream=True)
                total_length = response.headers.get('content-length')

                # Resetting for visualization
                self.data_length = 0
                self.total_length = int(total_length)

                if total_length is None: # no content length header
                    f.write(response.content)
                else:
                    try:
                        for data in response.iter_content(chunk_size=CHUNK_SIZE):
                            self._display_download_progress(len(data))
                            f.write(data)
                    except:
                        self.log.error(f"Failed downloading file from '{link}'")
        self.log.info(f"File saved to: '{output}'")

    def clone_git_repo(self, destination: Path, repo_url: str, checkout: str) -> None:
        '''
        Clones a git repo to a destination folder if it does not yet exist.
        
        Args:
            destination: The path to where the git repo should be cloned to.
            repo_url: The URL to do the git clone
            checkout: The tag, branch or commit hash to checkout
        '''
                
        if destination.is_dir():
            self.log.info(f"Folder: {destination} exists already. No 'git clone' is performed.")

        else:
            self.log.info(f"Clonging {repo_url} to {destination}")
            repo = Repo.clone_from(repo_url, destination)
            repo.git.checkout(checkout)
            self.log.info(f"Successfully cloned and checked out '{checkout}' of {repo_url}")

    def _display_download_progress(self, data_chunk_size: int) -> None:
        '''
        Display the download progress in console. 
        
        Args:
            data_chunk_size: Integer of size of the newly downloaded data chunk.
        '''
        self.data_length += data_chunk_size
        done = int(LOADING_BAR_LENGTH * self.data_length / self.total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (LOADING_BAR_LENGTH-done)) )    
        sys.stdout.flush()
import requests
import json
import pickle as pkl
import pandas as pd
from helical.utils.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from azure.storage.blob import  BlobClient
from azure.core.pipeline.transport import RequestsTransport
from git import Repo
from helical.constants.paths import CACHE_DIR_HELICAL

LOGGER = logging.getLogger(__name__)
INTERVAL = 1000 # interval to get gene mappings
CHUNK_SIZE = 1024 * 1024 * 10 #8192 # size of individual chunks to download
LOADING_BAR_LENGTH = 50 # size of the download progression bar in console
class Downloader(Logger):
    def __init__(self, loging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(loging_type, level)
        self.display = True

        # manually create a requests session
        self.session = requests.Session()
        # set an adapter with the required pool size
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=100,pool_connections=100)
        # mount the adapter to the session
        self.session.mount('https://', adapter)

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
            LOGGER.exception(f"Failed to open the '{path_to_ets_csv}' file. Please provide it.")

        if output.is_file():
            LOGGER.info(f"No mapping is done because mapping file already exists here: '{output}'")

        else:
            genes = df['egid'].dropna().unique()

            server = "https://rest.ensembl.org/lookup/id"
            headers={ "Content-Type" : "application/json", "Accept" : "application/json"}

            ensemble_to_display_name = dict()
            
            LOGGER.info(f"Starting to download the mappings of {len(genes)} genes from '{server}'")

            # Resetting for visualization
            self.data_length = 0
            self.total_length = len(genes)

            for i in range(0, len(genes), INTERVAL):
                if self.display: 
                    self._display_download_progress(INTERVAL)
                ids = {'ids':genes[i:i+INTERVAL].tolist()}
                r = requests.post(server, headers=headers, data=json.dumps(ids))
                decoded = r.json()
                ensemble_to_display_name.update(decoded)

            pkl.dump(ensemble_to_display_name, open(output, 'wb')) 
            LOGGER.info(f"Downloaded all mappings and saved to: '{output}'")

    def download_via_link(self, output: Path, link: str) -> None:
        '''
        Download a file via a link. 
        
        Args:
            output: Path to the output file.
            link: URL to download the file from.
        
        Raises:
            Exception: If the download fails.
        '''
       
        if output.is_file():
            LOGGER.info(f"File: '{output}' exists already. File is not overwritten and nothing is downloaded.")

        else:
            LOGGER.info(f"Starting to download: '{link}'")
            with open(output, "wb") as f:
                response = requests.get(link, stream=True)

                if response.status_code != 200:
                    message = f"Failed downloading file from '{link}' with status code: {response.status_code}"
                    LOGGER.error(message)
                    raise Exception
                
                total_length = response.headers.get('content-length')

                # Resetting for visualization
                self.data_length = 0
                self.total_length = int(total_length)

                if total_length is None: # no content length header
                    f.write(response.content)
                else:
                    try:
                        for data in response.iter_content(chunk_size=CHUNK_SIZE):
                            if self.display: 
                                self._display_download_progress(len(data))
                            f.write(data)
                    except:
                        LOGGER.error(f"Failed downloading file from '{link}'")
        LOGGER.info(f"File saved to: '{output}'")

    def clone_git_repo(self, destination: Path, repo_url: str, checkout: str) -> None:
        '''
        Clones a git repo to a destination folder if it does not yet exist.
        
        Args:
            destination: The path to where the git repo should be cloned to.
            repo_url: The URL to do the git clone
            checkout: The tag, branch or commit hash to checkout
        '''
                
        if destination.is_dir():
            LOGGER.info(f"Folder: {destination} exists already. No 'git clone' is performed.")

        else:
            LOGGER.info(f"Clonging {repo_url} to {destination}")
            repo = Repo.clone_from(repo_url, destination)
            repo.git.checkout(checkout)
            LOGGER.info(f"Successfully cloned and checked out '{checkout}' of {repo_url}")

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

    def download_via_name_v0(self, name: str) -> None:
        '''
        Download a file via a link. 
        
        Args:
            output: Path to the output file.
            link: URL to download the file from.
        '''
        main_link = "https://helicalpackage.blob.core.windows.net/helicalpackage/data"
        output = os.path.join(CACHE_DIR_HELICAL, name)

        link = f"{main_link}/{name}"
        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output),exist_ok=True)

        if Path(output).is_file():
            LOGGER.info(f"File: '{output}' exists already. File is not overwritten and nothing is downloaded.")

        else:
            LOGGER.info(f"Starting to download: '{link}'")
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
                        # for data in response.iter_content(chunk_size=CHUNK_SIZE):
                        pbar = tqdm(total=int(self.total_length), unit="B", unit_scale=True)
                        for data in tqdm(response.iter_content(chunk_size=CHUNK_SIZE)):
                            # self._display_download_progress(len(data))
                            f.write(data)
                            pbar.update(len(data))
                    except:
                        LOGGER.error(f"Failed downloading file from '{link}'")
        LOGGER.info(f"File saved to: '{output}'")


    def download_via_name(self, name: str) -> None:
        '''
        Download a file via a link. 
        
        Args:
            name (str): The name of the file to be downloaded.
        
        Returns:
            None
        '''

        main_link = "https://helicalpackage.blob.core.windows.net/helicalpackage/data"
        output = os.path.join(CACHE_DIR_HELICAL, name)

        blob_url = f"{main_link}/{name}"

        # Create a BlobClient object for the specified blob
        blob_client = BlobClient.from_blob_url(blob_url,max_single_get_size=1024*1024*32,max_chunk_get_size=1024*1024*4,session=self.session)
        

        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output),exist_ok=True)
            LOGGER.info(f"Creating Folder {os.path.dirname(output)}")

        if Path(output).is_file():
            LOGGER.info(f"File: '{output}' exists already. File is not overwritten and nothing is downloaded.")

        else:
            LOGGER.info(f"Starting to download: '{blob_url}'")
            # disabling logging info messages from Azure package as there are too many
            logging.disable(logging.INFO)
            self.display_azure_download_progress(blob_client, blob_url, output)
            logging.disable(logging.NOTSET)
            
        LOGGER.info(f"File saved to: '{output}'")

    def display_azure_download_progress(self, blob_client: BlobClient, blob_url: str, output: Path) -> None:
        """
        Displays the progress of an Azure blob download and saves the downloaded file.

        Args:
            blob_client (BlobClient): The BlobClient object used to download the blob.
            blob_url (str): The URL of the blob to be downloaded.
            output (Path): The path where the downloaded file will be saved.

        Returns:
            None
        """
        # Resetting for visualization
        self.data_length = 0
        total_length = blob_client.get_blob_properties().size

        # handle displaying download progress or not
        if self.display:
            pbar = tqdm(total=total_length, unit='B', unit_scale=True, desc='Downloading')
            def progress_callback(bytes_transferred, total_bytes):
                pbar.update(bytes_transferred-pbar.n)
        else:
            pbar = None
            def progress_callback(bytes_transferred, total_bytes):
                pass

        # actual download
        try:
            with open(output, "wb") as sample_blob:
                download_stream = blob_client.download_blob(max_concurrency=100,progress_hook=progress_callback)

                sample_blob.write(download_stream.readall())
        except:
            LOGGER.error(f"Failed downloading file from '{blob_url}'")
        
        if self.display:
            pbar.close()

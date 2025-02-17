import requests
from helical.utils.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from azure.storage.blob import  BlobClient
from helical.constants.paths import CACHE_DIR_HELICAL
import hashlib

LOGGER = logging.getLogger(__name__)
INTERVAL = 1000 # interval to get gene mappings
CHUNK_SIZE = 1024 * 1024 * 10 #8192 # size of individual chunks to download
LOADING_BAR_LENGTH = 50 # size of the download progression bar in console

HASH_DICT = {
    'uce/4layer_model.torch': '16430370e0d672c8db6e275440e7974d2fd0a21f29aa9299e141085f82a5a886',
    'uce/33l_8ep_1024t_1280.torch': 'aa6457a0eb2e91d8382d96fb455456e40a9423a00509ea296079a75b1a9390c0',
    'uce/all_tokens.torch': 'e3e3ad03a9f8fdca8babec5b0c72f7f4043a4bec2e3eb009b8fe1b28d984c93a',
    'uce/species_chrom.csv': '7f5d32e6adcc3786c613043a4de8e2a47187935cfb9a1d3fcf7373eb50caebf7',
    'uce/species_offsets.pkl': 'abda5b2bc4018187e408623b292686a061912f449daceb4c9c9603caf0d62538',
    'uce/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt': '06962edd34edaed111df9887845e73fa6e7ce3473ad71d99a54d66130c2b475e',
    'uce/protein_embeddings/Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt': 'b5821f00221a3c42956a9aee4779637144a0201503d4a0e201cee0ca04769986',
    'uce/protein_embeddings/Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt': '732baa76fb2e2bf887e913af0c98fb6e12d70cf4a11100061f334752ac03037d',
    'uce/protein_embeddings/Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt': '47520be81301881bee44733854523f500591c1c44afcac012587df4fb80c426a',
    'uce/protein_embeddings/Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt': '33c787398663b70f45b088e78db30ad3d599333bd7ad913c84ad8e6e098aceea',
    'uce/protein_embeddings/Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt': 'a9add6cd9acd4962c7e6843736d4e3eff2557f0b742018d0ff718128f231c40b',
    'uce/protein_embeddings/Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt': '7873ebc64f12de56492ac304232872533f02f3a7f8f28f1c60238a58224ebf16',
    'uce/protein_embeddings/Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt': '5d165156a7ed447fd52282ce6d215ec06bab693a7fa5c6901e7a2545eafff9dc',

    'scgpt/scGPT_CP/vocab.json': 'da58faa9151d9142573ac59568f831f7a6caa912d9c7f2311591878a30f13666',
    'scgpt/scGPT_CP/best_model.pt': 'fcdeff193dbf4d421a4d588ef1affa0b2612d33892b3281650c7cd1cea43ebc7',

    'geneformer/v1/gene_median_dictionary.pkl': 'b509e0d0227acf223c72ca4f604ce47a1f7af84eb1d53d1466c1c340103e9a2b',
    'geneformer/v1/token_dictionary.pkl': 'e2c77b0f292c3e3a98ae0a0b562d5feb6d31373c655cd3b4a61a0c748794de24',
    'geneformer/v1/ensembl_mapping_dict.pkl': '28391ff889e406ee580ace21eb40fb73072dd69a2059b5204f17c4efa2d3bbf0',
    'geneformer/v1/gf-6L-30M-i2048/config.json': '9cf69ca3bdb0215c4188b54c451b6f02adfe68b8f66011a57d0f32845133fd4b',
    'geneformer/v1/gf-6L-30M-i2048/training_args.bin': 'f0ec3459454205174c9d2e4d6c6930f6b0fbf3364fc03a6f4d99c4d3add2012b',
    'geneformer/v1/gf-6L-30M-i2048/pytorch_model.bin': '9dc411f9667850bd6bb76e9e8cf2f0b923d7501780fb4c277adae55965c476d5',
    'geneformer/v1/gf-12L-30M-i2048/config.json': 'a82b9cf2cb830be227bfbbe5a4c5d62723f49fac892ca37c541d0ef40a0e1de9',
    'geneformer/v1/gf-12L-30M-i2048/training_args.bin': '259cf6067211e24e198690d00f0a222ee5550ad57e23d04ced0d0ca2e1b3738e',
    'geneformer/v1/gf-12L-30M-i2048/pytorch_model.bin': 'cea6f8480b6267c3622d57b9c1d53d0f2fa4df38379cf64ec49a35e075afa09d',

    'geneformer/v2/gene_median_dictionary.pkl': '2be7704fd679a43720011fa0337a5a34d2cf3cb48768c656680dca3dd0653b75',
    'geneformer/v2/token_dictionary.pkl': '12b094814a5764310a2be428b81748fe0af1b246832384a2b187923481b93c8c',
    'geneformer/v2/ensembl_mapping_dict.pkl': '28391ff889e406ee580ace21eb40fb73072dd69a2059b5204f17c4efa2d3bbf0',

    'geneformer/v2/gf-20L-95M-i4096/training_args.bin': '5afed602918d6f0c4916c1b9335bcdb619bca2c6fd6c7e0dd2a86d195264b8cc',
    'geneformer/v2/gf-20L-95M-i4096/config.json': '915948b8161a15747647c9dd04d4bd3a950ca5fb145a14d9bc1157948a4cb9e7',
    'geneformer/v2/gf-20L-95M-i4096/generation_config.json': '07f46e7e174ffc98dd5072d6a7e0df8935f44046258d30dbf7b78edadcd44af4',
    'geneformer/v2/gf-20L-95M-i4096/model.safetensors': '5109b987c2e390b7bc46f77675bf020f94125ed36e2ba968b52cee7674106669',

    'geneformer/v2/gf-12L-95M-i4096-CLcancer/training_args.bin': '37074f3ea62a6ba0a312c38526c20c2dccbb068a2c7ee8c7c73b435dd90ab7b1',
    'geneformer/v2/gf-12L-95M-i4096-CLcancer/config.json': '7d3720eb553238849f7b3b3fd874e2bafb59c97b0e70afd7ca90132c43b8d5b1',
    'geneformer/v2/gf-12L-95M-i4096-CLcancer/generation_config.json': '07f46e7e174ffc98dd5072d6a7e0df8935f44046258d30dbf7b78edadcd44af4',
    'geneformer/v2/gf-12L-95M-i4096-CLcancer/model.safetensors': 'b5add9f834ee85a3ed10416b34acab3815f3f8f1b045d83274618f54b6667bb3',

    'geneformer/v2/gf-12L-95M-i4096/config.json': 'f56780389d8c89c1b6c4084e2e6ee1f736558e4b3bb8ce7473159e83465de401',
    'geneformer/v2/gf-12L-95M-i4096/training_args.bin': '21a45980734b138029422e95a5601def858821a9ec02cd473938b9f525ac108d',
    'geneformer/v2/gf-12L-95M-i4096/generation_config.json': '07f46e7e174ffc98dd5072d6a7e0df8935f44046258d30dbf7b78edadcd44af4',
    'geneformer/v2/gf-12L-95M-i4096/model.safetensors': '25e191c9b3e1762a967260894df12e6d6f0daf2c0dc6f8f0fa055ea77bf8c8ba',

    'hyena_dna/hyenadna-tiny-1k-seqlen.ckpt': 'dc6a481cbebe567ed6c68da856479dc69e66ea2f4cb0a59f6da16a75c91785d9',
    'hyena_dna/hyenadna-tiny-1k-seqlen-d256.ckpt': '381e95e5b1c6ad2ce19b4d00ef2bbadeb15548cad7401e07a4f17ea4953407f1',
    'hyena_dna/hyenadna-small-32k-seqlen.ckpt': '72af6f9bd5a04fcecc38a19c89eac3259baa8474b34252ed8faeeabceae020ab',
    'hyena_dna/hyenadna-medium-450k-seqlen.ckpt': '09f31b8de637b49c5c02bf2a31afc8d7440aa7456a1ba36718a7188293c5cf7d',
    'hyena_dna/hyenadna-large-1m-seqlen.ckpt': '35a17f681717df3881921a34f87c44bbf2786161830bf51fa0f9d9fd747b54d5',

    'caduceus/caduceus-ph-16L-seqlen-131k-d256/model.safetensors': '6769571c4a8ec30a758a89131e4288b5448989596e5c817cb3759280220a898d',
    'caduceus/caduceus-ph-16L-seqlen-131k-d256/config.json': 'a6c32e0f0d30f558a4d971db510f5c0f2584d9ba791034b8a693a021280ef6e0',
    'caduceus/caduceus-ph-4L-seqlen-1k-d118/model.safetensors': 'b1dbe9b683344327286efd400f10d686e5dca62cde93f99021d9fdfd0b78dd8a',
    'caduceus/caduceus-ph-4L-seqlen-1k-d118/config.json': '3ebd827c68a3a3f19ae96e41f61663f739ffa7b89fdb5e4ce4df606afbea5156',
    'caduceus/caduceus-ph-4L-seqlen-1k-d256/model.safetensors': '0dd7a9f3c161b99c6bec513786b11046b51dd5c64ba0ae207c6422edd236a358',
    'caduceus/caduceus-ph-4L-seqlen-1k-d256/config.json': 'a8eeceb556ec73131bc15268321275fd195f26dfa1fbee894d89b66fac67de9d',
    'caduceus/caduceus-ps-16L-seqlen-131k-d256/model.safetensors': '05398c433ebd4175412c8fc8c934e369eba55b4fa2ce555f4cc40610f156624c',
    'caduceus/caduceus-ps-16L-seqlen-131k-d256/config.json': '53be8604fa06d3389e088a777ba1e796f1893cefe57f0109acc45f1f0c361dfd',
    'caduceus/caduceus-ps-4L-seqlen-1k-d118/model.safetensors': 'fac204c3415c86de0a5d381ba49ad72570a63574364fbd175b2b38469fbf4928',
    'caduceus/caduceus-ps-4L-seqlen-1k-d118/config.json': '8185189b36e007b04f943157388500681c7499a8b264ffc4022cceb1db97fd66',
    'caduceus/caduceus-ps-4L-seqlen-1k-d256/model.safetensors': 'bbf5a7e1ffdfda9a31fa91999ccbe2ce8fec329ab4827c90d284de2d5851358c',
    'caduceus/caduceus-ps-4L-seqlen-1k-d256/config.json': '655d2c3a692ab35718cfe87ce98b14c4e57807f85089c2af81809873f356349f',

    'genept/genept_embeddings/genept_embeddings.json': '54a58177e6f4cb9c2d98f39cb8c586bd347a526375eba861df15a3714f737ccc',

    '17_04_24_YolkSacRaw_F158_WE_annots.h5ad': '0585c186ef23951a538522dd6882492c2d5c165c615543fe01bf0d0daedc2f5a',
}

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
            LOGGER.debug(f"File: '{output}' exists already. File is not overwritten and nothing is downloaded.")

        else:
            LOGGER.info(f"Starting to download: '{link}'")
            response = requests.get(link, stream=True)

            if response.status_code != 200:
                message = f"Failed downloading file from '{link}' with status code: {response.status_code}"
                LOGGER.error(message)
                raise Exception(message)
            
            total_length = response.headers.get('content-length')

            # Resetting for visualization
            self.data_length = 0
            self.total_length = int(total_length)

            with open(output, "wb") as f:
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

    def calculate_partial_file_hash(self, file_path: str, chunk_size: int = 8192, algorithm: str = 'sha256') -> str:
        """
        Calculate the hash of the first and last chunks of a file.
        
        Args:
            file_path (str): Path to the file.
            chunk_size (int): Size of the chunks to hash (in bytes).
            algorithm (str): Hashing algorithm (e.g., 'sha256', 'blake2b', 'md5'), default is 'sha256'.
        
        Returns:
            str: Combined hash of the first and last chunks.
        """
        hash_func = hashlib.new(algorithm)
        file_size = os.path.getsize(file_path)

        with open(file_path, 'rb') as f:
            # Read the first chunk
            first_chunk = f.read(chunk_size)
            hash_func.update(first_chunk)
            
            # Read the last chunk
            if file_size > chunk_size:
                f.seek(-chunk_size, os.SEEK_END)
                last_chunk = f.read(chunk_size)
                hash_func.update(last_chunk)
        
        return hash_func.hexdigest()

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

        if Path(output).is_file() and self.calculate_partial_file_hash(output) == HASH_DICT[name]:
            LOGGER.debug(f"File: '{output}' exists already. File is not overwritten and nothing is downloaded.")

        else:
            LOGGER.info(f"File does not exist or has incorrect hash. Starting to download: '{blob_url}'")
            # disabling logging info messages from Azure package as there are too many
            logging.disable(logging.INFO)
            self.display_azure_download_progress(blob_client, blob_url, output)
            logging.disable(logging.NOTSET)
            assert self.calculate_partial_file_hash(output) == HASH_DICT[name], f"Hash of downloaded file '{output}' does not match the expected hash."
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

import os
from typing import Any, Literal

import anndata as ad
from scanpy.readwrite import _check_datafile_present_and_download

PathLike = os.PathLike | str


def tabula_sapiens(
    tissue: Literal[
        "lymphnode",
        "heart",
        "ear",
        "endothelium",
        "epithelium",
        "germline",
        "immune",
        "neural",
        "stromal",
        "bladder",
        "blood",
        "bone_marrow",
        "ear",
        "eye",
        "fat",
        "heart",
        "kidney",
        "large_intestine",
        "liver",
        "lung",
        "lymph_node",
        "mammary",
        "muscle",
        "ovary",
        "pancreas",
        "prostate",
        "salivary_gland",
        "skin",
        "small_intestine",
        "spleen",
        "stomach",
        "testis",
        "thymus",
        "tongue",
        "trachea",
        "uterus",
        "vasculature",
    ],
    path: PathLike | None = None,
    force_download: bool = False,
    version: Literal["v1", "v2"] = "v2",
    **kwargs: Any,
) -> ad.AnnData:
    """
    Load tissue dataset from Tabula Sapiens.

    Args:
        tissue: Tissue name e.g. ('lymphnode', 'heart', or 'ear')
        path: Path to save the dataset. If None, uses default path.
        force_download: Whether to force download the dataset.
        version: Version of the dataset. 'v1' is in-distribution for Transcriptformer,
                'v2' is out-of-distribution.

    Returns
    -------
        AnnData object.
    """
    urls = {
        "endothelium": "https://datasets.cellxgene.cziscience.com/de698978-3267-45c6-b492-4b0636ef564d.h5ad",
        "epithelium": "https://datasets.cellxgene.cziscience.com/f4f99331-40cf-4bc7-bbb5-c4dc4c150840.h5ad",
        "germline": "https://datasets.cellxgene.cziscience.com/d81eae50-e96d-44a6-876a-540a3a4610dd.h5ad",
        "immune": "https://datasets.cellxgene.cziscience.com/7f2e355e-9944-4477-98f0-2184943d9d2c.h5ad",
        "neural": "https://datasets.cellxgene.cziscience.com/74d7bc5e-ac01-43d6-ae70-8ffe82b8afde.h5ad",
        "stromal": "https://datasets.cellxgene.cziscience.com/1c63b188-d097-441f-8922-b7ebaf715cca.h5ad",
        "bladder": "https://datasets.cellxgene.cziscience.com/01ef7ded-020c-4292-b253-a234f7331757.h5ad",
        "blood": "https://datasets.cellxgene.cziscience.com/4eb58518-23ce-43e8-89ca-9e9fa87c081d.h5ad",
        "bone_marrow": "https://datasets.cellxgene.cziscience.com/bbf12ff4-7f22-4cf5-83cf-9cd37ce4d6f4.h5ad",
        "ear": "https://datasets.cellxgene.cziscience.com/51233375-325e-4e8e-b05d-208fd8cc4ba4.h5ad",
        "eye": "https://datasets.cellxgene.cziscience.com/a77c03ba-0748-4382-94bd-e1c358175d5a.h5ad",
        "fat": "https://datasets.cellxgene.cziscience.com/4ccbfe8d-0e59-4433-b53c-55a717a32d16.h5ad",
        "heart": "https://datasets.cellxgene.cziscience.com/832cf70d-ffdd-48c1-b2cc-ffbb30b42e6c.h5ad",
        "kidney": "https://datasets.cellxgene.cziscience.com/b7581463-9676-46fd-9b84-73132109b2a4.h5ad",
        "large_intestine": "https://datasets.cellxgene.cziscience.com/8a18fafb-9d37-4bae-9a18-00e35368ca23.h5ad",
        "liver": "https://datasets.cellxgene.cziscience.com/c264e09f-7c3b-4294-b0f4-82a790bd0014.h5ad",
        "lung": "https://datasets.cellxgene.cziscience.com/7c5ee207-950a-4ae3-bd85-2483f75111d7.h5ad",
        "lymph_node": "https://datasets.cellxgene.cziscience.com/b439da44-6128-483f-9424-10c5618a8b7e.h5ad",
        "mammary": "https://datasets.cellxgene.cziscience.com/c207bca6-a5c8-472e-be14-83b7b8cc83a6.h5ad",
        "muscle": "https://datasets.cellxgene.cziscience.com/9bea986b-f7f1-479b-92d9-61a893629399.h5ad",
        "ovary": "https://datasets.cellxgene.cziscience.com/6e4ccce5-532e-401e-b1bf-ce01f4883eff.h5ad",
        "pancreas": "https://datasets.cellxgene.cziscience.com/240e2b84-6893-4651-830b-5baebe8b8bfe.h5ad",
        "prostate": "https://datasets.cellxgene.cziscience.com/250cb362-68b4-426e-b4b6-1f7c478197d2.h5ad",
        "salivary_gland": "https://datasets.cellxgene.cziscience.com/ded561fb-8668-4ed9-b239-f03a01584dda.h5ad",
        "skin": "https://datasets.cellxgene.cziscience.com/c458ba64-399b-4afb-b4b9-8ac91b8f43b3.h5ad",
        "small_intestine": "https://datasets.cellxgene.cziscience.com/a77da5e7-8a67-4351-9ec3-49a9c80060ab.h5ad",
        "spleen": "https://datasets.cellxgene.cziscience.com/79dd0749-aba8-40b8-80b8-2937377c2bdd.h5ad",
        "stomach": "https://datasets.cellxgene.cziscience.com/70b2a188-2183-4b20-80e2-8c870e6b01f0.h5ad",
        "testis": "https://datasets.cellxgene.cziscience.com/7ffe124c-b1ac-4998-932c-43e2038aacf9.h5ad",
        "thymus": "https://datasets.cellxgene.cziscience.com/3c5738bc-da1e-4b98-a2a6-685a7748e747.h5ad",
        "tongue": "https://datasets.cellxgene.cziscience.com/9e6d1c2e-b77d-4b36-b747-fad671e245f0.h5ad",
        "trachea": "https://datasets.cellxgene.cziscience.com/f950a2b4-0761-4ec3-b74e-4d951e2fe30b.h5ad",
        "uterus": "https://datasets.cellxgene.cziscience.com/e53087ec-74a9-4a37-b5fb-36911c3a68c7.h5ad",
        "vasculature": "https://datasets.cellxgene.cziscience.com/866ad7f6-bf26-4da4-b078-1b5eebee0fd6.h5ad",
    }

    if path is None:
        path = f"~/.cache/transcriptformer/{tissue}_{version}.h5ad"

    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url=urls[tissue],
        force_download=force_download,
        **kwargs,
    )

    return filter_anndata_by_tissue_and_version(adata, version=version)


def bgee_testis_evolution(
    organism: Literal[
        "marmoset",
        "rhesus_macaque",
        "human",
        "chimpanzee",
        "platypus",
        "mouse",
        "opossum",
        "gorilla",
        "chicken",
    ],
    path: PathLike | None = None,
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """
    Load testis tissue from an organism dataset from Bgee Evolution.

    Args:
        organism: Organism name e.g. ('marmoset', 'rhesus_macaque').
        path: Path to save the dataset. If None, uses default path.
        force_download: Whether to force download the dataset.

    Returns
    -------
        AnnData object.
    """
    urls = {
        "marmoset": "datasets/v1/evo_distance/testis/Callithrix_jacchus_ERP132588_droplet_based_curated.h5ad",
        "chicken": "datasets/v1/evo_distance/testis/Gallus_gallus_ERP132576_droplet_based_curated.h5ad",
        "gorilla": "datasets/v1/evo_distance/testis/Gorilla_gorilla_ERP132581_droplet_based_curated.h5ad",
        "human": "datasets/v1/evo_distance/testis/Homo_sapiens_ERP132584_droplet_based_curated.h5ad",
        "macaque": "datasets/v1/evo_distance/testis/Macaca_mulatta_ERP132582_droplet_based_curated.h5ad",
        "opossum": "datasets/v1/evo_distance/testis/Monodelphis_domestica_ERP132579_droplet_based_curated.h5ad",
        "mouse": "datasets/v1/evo_distance/testis/Mus_musculus_ERP132577_droplet_based_curated.h5ad",
        "platypus": "datasets/v1/evo_distance/testis/Ornithorhynchus_anatinus_ERP132578_droplet_based_curated.h5ad",
        "chimpanzee": "datasets/v1/evo_distance/testis/Pan_troglodytes_ERP139683_droplet_based_curated.h5ad",
    }

    if path is None:
        path = f"~/.cache/transcriptformer/{organism}.h5ad"

    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url=urls[organism],
        bucket_name="cz-benchmarks-data",
        force_download=force_download,
        **kwargs,
    )

    return adata


def download_all_embeddings(
    path: PathLike | None = None,
    force_download: bool = False,
    **kwargs: Any,
) -> None:
    """
    Download all embeddings used in the Transcriptformer paper.

    Args:
        path: Path to save the dataset. If None, uses default path.
        force_download: Whether to force download the dataset and overwrite existing files.
        **kwargs: Additional arguments passed to the download function.
    """
    import shutil
    import tarfile

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    file_name = "all_embeddings.tar.gz"

    if path is None:
        path = "~/.cache/transcriptformer"

    # Ensure path is expanded and normalized
    path = os.path.expanduser(path)

    # Define the tar file path and extraction directory
    tar_path = os.path.join(path, file_name)
    extraction_dir = path  # Extract to the same directory as the tar file

    bucket_name = "czi-transcriptformer"
    backup_url = "weights/" + file_name
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # If force_download is True, remove existing files
    if force_download:
        # Remove extracted files if they exist
        for item in os.listdir(extraction_dir):
            item_path = os.path.join(extraction_dir, item)
            if (
                os.path.isdir(item_path)
                and item != os.path.basename(file_name).split(".")[0]
            ):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path) and item != file_name:
                os.remove(item_path)

        # Remove tar file if it exists
        if os.path.exists(tar_path):
            os.remove(tar_path)

    # Download the file if it doesn't exist
    if not os.path.exists(tar_path):
        s3_client.download_file(bucket_name, backup_url, tar_path)

    # Extract the tar file
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extraction_dir, member.name))
            if not member_path.startswith(os.path.abspath(extraction_dir)):
                raise ValueError(f"Illegal tar archive entry: {member.name}")
            if member.islnk() or member.issym():
                raise ValueError(
                    f"Unsupported symbolic link in tar archive: {member.name}"
                )
        tar.extractall(
            extraction_dir,
            members=[
                member
                for member in tar.getmembers()
                if os.path.abspath(
                    os.path.join(extraction_dir, member.name)
                ).startswith(os.path.abspath(extraction_dir))
            ],
        )


def _load_dataset_from_url(
    fpath: PathLike,
    file_type: Literal["h5ad"],
    *,
    backup_url: str,
    bucket_name: str | None = None,
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    fpath = os.path.expanduser(fpath)
    assert file_type in [
        "h5ad"
    ], f"Invalid type `{file_type}`. Must be one of `['h5ad']`."
    if not fpath.endswith(file_type):
        fpath += f".{file_type}"
    if force_download and os.path.exists(fpath):
        os.remove(fpath)
    if bucket_name is not None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        if not os.path.exists(fpath):
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            # Download the file
            s3_client.download_file(bucket_name, backup_url, fpath)
    else:
        if not _check_datafile_present_and_download(backup_url=backup_url, path=fpath):
            raise FileNotFoundError(f"File `{fpath}` not found or download failed.")
    data = ad.read_h5ad(filename=fpath, **kwargs)

    return data


def filter_anndata_by_tissue_and_version(
    adata: ad.AnnData,
    version: Literal["v1", "v2"],
    min_gene_counts: int = 10,
) -> ad.AnnData:
    if version == "v1":
        mask = adata.obs["donor_id"].str.split("TSP").str[-1].astype(int) < 16
    elif version == "v2":
        mask = adata.obs["donor_id"].str.split("TSP").str[-1].astype(int) > 16
    else:
        raise ValueError(f"Invalid version: {version}. Must be one of ['v1', 'v2']")

    adata_filtered = adata[mask].copy()

    gene_sums = adata_filtered.X.sum(axis=0)
    genes_to_keep = gene_sums >= min_gene_counts
    adata_filtered = adata_filtered[:, genes_to_keep].copy()

    return adata_filtered


if __name__ == "__main__":
    download_all_embeddings()

import pytest
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from helical.utils import get_anndata_from_hf_dataset

def create_mock_dataset(gene_names: str):
    data = {
        'raw_counts': [
            [1, 2, 3, 2],
            [98],
            [72, 19]
        ],
        'rows': [
            [0, 1, 2, 3],
            [0],
            [1, 3]
        ],
        'obs1': [10, 20, 30],
        'obs2': [40, 50, 60],
        'size': [4, 4, 4]
    }
    features = Features({
        'raw_counts': Sequence(Value('uint32'), -1, gene_names),
        'rows': Sequence(Value('uint32')),
        'obs1': Value('int64'),
        'obs2': Value('int64'),
        'size': Value('uint32')
    })
    dataset = Dataset.from_dict(data, features=features)
    return dataset

def test_get_anndata_from_hf_dataset():
    dataset = create_mock_dataset("gene1,gene2,gene3,gene4")
    ann_data = get_anndata_from_hf_dataset(dataset)
    
    assert ann_data.shape == (3, 4)

    # assert that observation names are correct (ie. no 'rows', 'raw_counts', or 'size')
    assert list(ann_data.obs.columns) == ['obs1', 'obs2']

    # assert that gene names are converted to uppercase 
    assert list(ann_data.var_names) == ['GENE1', 'GENE2', 'GENE3', 'GENE4']
    assert list(ann_data.var['gene_name']) == ['GENE1', 'GENE2', 'GENE3', 'GENE4']
    
    # assert that counts are placed in the correct positions
    assert np.array_equal(ann_data.X.toarray(), np.array([[1, 2, 3, 2], [98, 0, 0, 0], [0, 72, 0, 19]]))

def test_get_anndata_from_hf_dataset_mismatched_gene_names():
    dataset = create_mock_dataset("gene1,gene2")
    
    with pytest.raises(ValueError):
        get_anndata_from_hf_dataset(dataset)
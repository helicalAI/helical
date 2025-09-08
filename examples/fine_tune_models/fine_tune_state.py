
from helical.models.state import stateFineTuningModel, stateConfig
import scanpy as sc
from omegaconf import DictConfig
import hydra
import os

@hydra.main(
    version_base=None, config_path="../run_models/configs", config_name="state_config"
)
def run_fine_tuning(cfg: DictConfig):
    # Get device parameter
    device = cfg.get("device", "cuda")
    
    # Load the desired dataset with error handling
    data_path = "competition_support_set/competition_val_template.h5ad"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please check the path.")
        return
    
    try:
        adata = sc.read_h5ad(data_path)
        print(f"Loaded data with shape: {adata.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Use only first 10 samples for testing
    adata = adata[:10].copy()
    print(f"Using subset with shape: {adata.shape}")

    # Get the desired label class
    if "cell_type" not in adata.obs.columns:
        print("Error: 'cell_type' column not found in adata.obs")
        return
        
    cell_types = list(adata.obs.cell_type)
    print(f"Found {len(cell_types)} cell types")

    # Get unique labels
    label_set = set(cell_types)
    print(f"Unique cell types: {label_set}")

    # Create the fine-tuning model with the relevant configs
    config = stateConfig(
        batch_size=8,
        model_dir="competition/first_run",
        model_config="configs/config.yaml",
        freeze_backbone=False
    )
        
    try:
        model = stateFineTuningModel(
            configurer=config, 
            fine_tuning_head="classification", 
            output_size=len(label_set),
            freeze_backbone=False
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Process the data for training
    try:
        dataset = model.process_data(adata)
        print("Data processed successfully")
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    # Create a dictionary mapping the classes to unique integers for training
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    print(f"Class mapping: {class_id_dict}")

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    print(f"Converted {len(cell_types)} labels to integers")

    # Fine-tune
    try:
        print("Starting fine-tuning...")
        model.train(train_input_data=dataset, train_labels=cell_types, epochs=1)
        print("Fine-tuning completed successfully")
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return

if __name__ == "__main__":
    run_fine_tuning()
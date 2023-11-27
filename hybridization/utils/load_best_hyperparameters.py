import os
from pathlib import Path

from Recommenders.DataIO import DataIO

def load_best_hyperparameters(folder: Path):
    """Load the best hyperparameters from a folder.

    Args:
        folder (Path): The folder Path object.

    Raises:
        RuntimeError: If no metadata file is found in the folder.

    Returns:
        Dict[str, Any]: the hyperparameters.
    """
    print(f"Loading best hyperparameters from {str(folder)}")
    metadata_file: str = None
    for f in folder.iterdir():
        fname = f.name
        if os.path.isfile(f) and fname.endswith("_metadata.zip"):
            metadata_file = f.name
            break
    
    if metadata_file is None:
        raise RuntimeError(f"No metadata file in {str(folder)}")

    data_loader = DataIO(folder_path=str(folder))
    search_metadata = data_loader.load_data(str(metadata_file))
    
    return search_metadata["hyperparameters_best"]
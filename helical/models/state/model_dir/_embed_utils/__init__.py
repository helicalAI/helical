import sys 

def _setup_vci_aliases():
    current_module = sys.modules[__name__]
    sys.modules["vci"] = current_module  # Only creates top-level vci

_setup_vci_aliases()
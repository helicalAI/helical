# Prerequisites

Have the `21iT009_051_full_data.csv` file in a base repo.
Then install the package `helical-package` like so

```
pip install git+https://github.com/helicalAI/helical-package.git
```

Copy the contents below in a python file `helical-test-package.py`
```
from helical.preprocessor import Preprocessor 
Preprocessor().save_ensemble_mapping('./21iT009_051_full_data.csv', './ensemble_to_display_name_batch_macaca.pkl')
```

Finally, execute like so:
```
python helical-test-package.py
```
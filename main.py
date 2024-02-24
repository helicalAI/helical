from helical.preprocessor import Preprocessor
from helical.constants.enums import LoggingLevel, LoggingType

if __name__ == "__main__":
    preprocessor = Preprocessor(LoggingType.CONSOLE, LoggingLevel.INFO)
    preprocessor.save_ensemble_mapping('./21iT009_051_full_data.csv', './ensemble_to_display_name_batch_macaca.pkl')
    

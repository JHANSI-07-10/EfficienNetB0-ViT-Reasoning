import os
import sys

# Add the current src directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Use direct imports so they work whether run directly or as a package
try:
    from .model_def import HybridSkinModel
    from .dataset import HAM10000
    from .predict import Predictor
    from .evaluate import evaluate
    from .utils import get_data_splits
except ImportError:
    from model_def import HybridSkinModel
    from dataset import HAM10000
    from predict import Predictor
    from evaluate import evaluate
    from utils import get_data_splits

__all__ = ["HybridSkinModel", "HAM10000", "Predictor", "evaluate", "get_data_splits"]
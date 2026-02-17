import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might be missing or slow to import
sys.modules['fiftyone'] = MagicMock()
sys.modules['fiftyone.brain'] = MagicMock()
sys.modules['fiftyone.zoo'] = MagicMock()
sys.modules['timm'] = MagicMock()
# sys.modules['torch'] = MagicMock() # Torch is needed for tensor checks if used

import torch
# Now import the classes under test
from visualization.Visualization import Visualizer
from methods.SelectionMethod import SelectionMethod

class TestRefactoring(unittest.TestCase):
    def test_sanitize_key(self):
        """Test _sanitize_key robustness."""
        # Test valid key
        key = Visualizer._sanitize_key("valid_key")
        self.assertEqual(key, "valid_key")
        
        # Test sanitization
        key = Visualizer._sanitize_key("invalid!@#key")
        self.assertEqual(key, "invalid___key")
        
        # Test empty key fallback
        key = Visualizer._sanitize_key("")
        self.assertEqual(key, "unnamed_run")

    @patch('methods.SelectionMethod.models')
    @patch('methods.SelectionMethod.data')
    def test_selection_method_methods(self, mock_data, mock_models):
        """Test that SelectionMethod has the new method and not old ones."""
        config = {
            'networks': {'type': 'ResNet18', 'params': {}},
            'num_gpus': 0,
            'training_opt': {'num_epochs': 10, 'num_data_workers': 0, 'batch_size': 32, 'loss_type': 'ce', 'resume': None},
            'dataset': {'name': 'cifar10'},
            'visualization': {'enable': True, 'milestones': [0.5], 'embedding_methods': ['umap']},
            'logger_opt': {'print_iter': 10},
            'methods': ['SelectionMethod']
        }
        logger = MagicMock()
        
        # Mock specific data return
        mock_data.cifar10.return_value = {
            'num_classes': 10,
            'train_dset': MagicMock(),
            'test_loader': MagicMock(),
            'num_train_samples': 100
        }
        
        # Instantiate
        # We need to mock create_optimizer etc which are imported from .method_utils
        # Since SelectionMethod imports method_utils.*, we can patch them where they are used?
        # Or patch methods.SelectionMethod.create_optimizer
        with patch('methods.SelectionMethod.create_optimizer') as mock_opt, \
             patch('methods.SelectionMethod.create_scheduler') as mock_sched, \
             patch('methods.SelectionMethod.create_criterion') as mock_crit:
            
            sm = SelectionMethod(config, logger)
            
            # Check methods presence
            self.assertTrue(hasattr(sm, 'process_milestones'), "process_milestones should exist")
            self.assertFalse(hasattr(sm, 'add_milestone_prediction_runs'), "add_milestone_prediction_runs should be improved/removed")
            self.assertFalse(hasattr(sm, 'tag_hardest_samples'), "tag_hardest_samples should be removed")

if __name__ == '__main__':
    unittest.main()

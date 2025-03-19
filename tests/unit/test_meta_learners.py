import pytest
import torch
from metalearn.core.algorithms.meta_learners import EMCC
from metalearn.core.models.networks import BayesianLSTMEncoder

@pytest.fixture
def sample_model():
    return BayesianLSTMEncoder(input_dim=10)

@pytest.fixture
def sample_tasks():
    return [{
        'support': torch.randn(10, 10),
        'query': torch.randn(5, 10)
    } for _ in range(5)]

def test_emcc_initialization(sample_model):
    learner = EMCC(sample_model)
    assert hasattr(learner, 'context_encoder')
    assert len(learner.elasticity) == len(list(sample_model.parameters()))

def test_emcc_meta_update(sample_model, sample_tasks):
    learner = EMCC(sample_model)
    optimizer = torch.optim.Adam(learner.parameters())
    
    try:
        metrics = learner.meta_update(sample_tasks, optimizer)
        assert 'total_loss' in metrics
        assert metrics['total_loss'] > 0
    except RuntimeError as e:
        pytest.fail(f"Meta update failed with error: {e}")
import pytest
from metalearn.core import EMCC, TaskBuilder
from metalearn.utils import DeviceManager

@pytest.fixture(scope="module")
def full_pipeline():
    manager = DeviceManager()
    model = BayesianLSTMEncoder(input_dim=10).to(manager.device)
    learner = EMCC(model)
    task_builder = TaskBuilder(...)
    return learner, task_builder, manager

def test_end_to_end_training(full_pipeline):
    learner, task_builder, manager = full_pipeline
    tasks = task_builder.generate_task_batch(10, steps=range(100))
    
    try:
        metrics = learner.meta_train(tasks)
        assert 'total_loss' in metrics
        assert metrics['total_loss'] > 0
    except RuntimeError as e:
        pytest.fail(f"Full pipeline failed: {e}")

def test_model_serialization(full_pipeline):
    learner, _, manager = full_pipeline
    test_input = torch.randn(5, 10).to(manager.device)
    
    # Test prediction consistency
    before = learner.predict(test_input)
    state = learner.save_model("test.pth")
    learner.load_model("test.pth")
    after = learner.predict(test_input)
    
    assert torch.allclose(before, after, atol=1e-6)
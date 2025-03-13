"""Basic tests for DMBD functionality."""

import pytest
import numpy as np
import torch
import os
import gc
from pathlib import Path

@pytest.fixture
def dmbd_model():
    """Fixture to create a basic DMBD model for testing."""
    try:
        from dmbd import DMBD
        
        # Basic model parameters
        obs_shape = (10, 10)
        role_dims = (4, 4, 4)
        hidden_dims = (3, 3, 3)
        
        model = DMBD(
            obs_shape=obs_shape,
            role_dims=role_dims,
            hidden_dims=hidden_dims,
            number_of_objects=1
        )
        
        return model
    except ImportError as e:
        pytest.skip(f"DMBD import failed: {e}")

def test_dmbd_import():
    """Test that DMBD can be imported."""
    try:
        from dmbd import DMBD
        assert True
    except ImportError as e:
        pytest.skip(f"DMBD import failed: {e}")

def test_dmbd_initialization(random_seed):
    """Test DMBD model initialization with basic parameters."""
    try:
        from dmbd import DMBD
        
        # Basic model parameters
        obs_shape = (10, 10)
        role_dims = (4, 4, 4)
        hidden_dims = (3, 3, 3)
        
        model = DMBD(
            obs_shape=obs_shape,
            role_dims=role_dims,
            hidden_dims=hidden_dims,
            number_of_objects=1
        )
        
        assert model is not None, "DMBD model initialization failed"
        assert hasattr(model, 'role_dims'), "Model missing role_dims attribute"
        assert hasattr(model, 'hidden_dims'), "Model missing hidden_dims attribute"
        assert hasattr(model, 'obs_shape'), "Model missing obs_shape attribute"
        
        # Check parameter shapes
        for name, param in model.named_parameters():
            assert param.ndim > 0, f"Parameter {name} should not be scalar"
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN values"
            assert not torch.isinf(param).any(), f"Parameter {name} contains infinite values"
        
    except ImportError as e:
        pytest.skip(f"DMBD import failed: {e}")

def test_dmbd_forward_pass(dmbd_model, random_seed, torch_device):
    """Test DMBD forward pass with dummy data."""
    try:
        # Create dummy data
        batch_size = 2
        seq_length = 5
        data = torch.randn(batch_size, seq_length, *dmbd_model.obs_shape, device=torch_device)
        
        # Move model to device
        dmbd_model = dmbd_model.to(torch_device)
        
        # Forward pass
        try:
            output = dmbd_model(data)
            assert output is not None, "Forward pass returned None"
            
            # Check output properties
            if hasattr(output, 'latent_states'):
                assert output.latent_states.shape[0] == batch_size, "Wrong batch size in output"
                assert output.latent_states.shape[1] == seq_length, "Wrong sequence length in output"
            
            # Check memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_used = torch.cuda.memory_allocated()
                print(f"GPU memory used: {memory_used / 1024**2:.2f} MB")
            
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
            
    except Exception as e:
        pytest.fail(f"Test setup failed: {e}")

def test_dmbd_loss_computation(dmbd_model, random_seed, torch_device):
    """Test DMBD loss computation with dummy data."""
    try:
        # Create dummy data
        batch_size = 2
        seq_length = 5
        data = torch.randn(batch_size, seq_length, *dmbd_model.obs_shape, device=torch_device)
        
        # Move model to device
        dmbd_model = dmbd_model.to(torch_device)
        
        # Forward pass and loss computation
        try:
            output = dmbd_model(data)
            loss = dmbd_model.loss(data, output)
            
            assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
            assert loss.ndim == 0, "Loss should be a scalar"
            assert not torch.isnan(loss), "Loss should not be NaN"
            assert not torch.isinf(loss), "Loss should not be infinite"
            
            # Test gradient computation
            loss.backward()
            
            # Check gradients
            for name, param in dmbd_model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Parameter {name} has no gradient"
                    assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
                    assert not torch.isinf(param.grad).any(), f"Parameter {name} has infinite gradients"
            
        except Exception as e:
            pytest.fail(f"Loss computation failed: {e}")
            
    except Exception as e:
        pytest.fail(f"Test setup failed: {e}")

def test_dmbd_parameter_shapes(dmbd_model):
    """Test that DMBD model parameters have correct shapes."""
    try:
        # Check parameter shapes
        for name, param in dmbd_model.named_parameters():
            assert param.ndim > 0, f"Parameter {name} should not be scalar"
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN values"
            assert not torch.isinf(param).any(), f"Parameter {name} contains infinite values"
            
        # Check specific parameter shapes
        total_role_dims = sum(dmbd_model.role_dims)
        total_hidden_dims = sum(dmbd_model.hidden_dims)
        
        # Verify role dimensions
        assert len(dmbd_model.role_dims) == 3, "Should have 3 role dimensions (s, b, z)"
        assert len(dmbd_model.hidden_dims) == 3, "Should have 3 hidden dimensions (s, b, z)"
        
        # Verify observation model dimensions
        if hasattr(dmbd_model, 'obs_model'):
            obs_shape = dmbd_model.obs_shape
            assert len(obs_shape) == 2, "Observation shape should be 2D"
            
    except Exception as e:
        pytest.fail(f"Parameter shape check failed: {e}")

def test_dmbd_multiple_objects(random_seed, torch_device):
    """Test DMBD with multiple objects."""
    try:
        from dmbd import DMBD
        
        # Model parameters
        obs_shape = (10, 10)
        role_dims = (4, 4, 4)
        hidden_dims = (3, 3, 3)
        num_objects = 2
        
        # Create model
        model = DMBD(
            obs_shape=obs_shape,
            role_dims=role_dims,
            hidden_dims=hidden_dims,
            number_of_objects=num_objects
        ).to(torch_device)
        
        # Create dummy data
        batch_size = 2
        seq_length = 5
        data = torch.randn(batch_size, seq_length, *obs_shape, device=torch_device)
        
        # Forward pass
        try:
            output = model(data)
            assert output is not None, "Forward pass returned None"
            
            # Check that the output contains information for multiple objects
            if hasattr(output, 'object_states'):
                assert output.object_states.shape[2] == num_objects, \
                    f"Expected {num_objects} objects in output"
                
            # Test object-specific operations
            for obj_idx in range(num_objects):
                # Get object-specific states if available
                if hasattr(output, 'object_states'):
                    obj_state = output.object_states[..., obj_idx, :]
                    assert obj_state is not None, f"State for object {obj_idx} is None"
                    assert not torch.isnan(obj_state).any(), f"NaN values in object {obj_idx} state"
            
        except Exception as e:
            pytest.fail(f"Multiple objects test failed: {e}")
            
    except ImportError as e:
        pytest.skip(f"DMBD import failed: {e}")

def test_dmbd_save_load(dmbd_model, random_seed, tmp_path):
    """Test saving and loading DMBD model."""
    try:
        # Create a temporary directory for saving
        save_dir = tmp_path / "dmbd_test"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        save_path = save_dir / "model.pt"
        torch.save(dmbd_model.state_dict(), save_path)
        
        # Create new model and load state
        loaded_model = type(dmbd_model)(
            obs_shape=dmbd_model.obs_shape,
            role_dims=dmbd_model.role_dims,
            hidden_dims=dmbd_model.hidden_dims,
            number_of_objects=1
        )
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            dmbd_model.named_parameters(), loaded_model.named_parameters()
        ):
            assert torch.allclose(param1, param2), \
                f"Parameter {name1} differs after loading"
        
        # Test with some data to ensure both models behave the same
        batch_size = 2
        seq_length = 5
        data = torch.randn(batch_size, seq_length, *dmbd_model.obs_shape)
        
        with torch.no_grad():
            output1 = dmbd_model(data)
            output2 = loaded_model(data)
            
            # Compare outputs
            if hasattr(output1, 'latent_states') and hasattr(output2, 'latent_states'):
                assert torch.allclose(output1.latent_states, output2.latent_states), \
                    "Model outputs differ after loading"
        
    except Exception as e:
        pytest.fail(f"Save/load test failed: {e}")

def test_dmbd_memory_cleanup(dmbd_model, random_seed, torch_device):
    """Test DMBD memory management and cleanup."""
    try:
        # Record initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Create and process some data
        batch_size = 2
        seq_length = 5
        data = torch.randn(batch_size, seq_length, *dmbd_model.obs_shape, device=torch_device)
        
        # Move model to device
        dmbd_model = dmbd_model.to(torch_device)
        
        # Run forward and backward passes
        output = dmbd_model(data)
        loss = dmbd_model.loss(data, output)
        loss.backward()
        
        # Clean up
        del output
        del loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            
            # Check memory usage
            memory_diff = final_memory - initial_memory
            print(f"Memory difference after cleanup: {memory_diff / 1024**2:.2f} MB")
            
            # Allow for some small residual memory usage
            assert memory_diff < 1024**2, "Significant memory not freed after cleanup"
        
    except Exception as e:
        pytest.fail(f"Memory cleanup test failed: {e}")

def test_dmbd_gradient_flow(dmbd_model, random_seed, torch_device):
    """Test gradient flow through the DMBD model."""
    try:
        # Create optimizer
        optimizer = torch.optim.Adam(dmbd_model.parameters(), lr=0.01)
        
        # Create dummy data
        batch_size = 2
        seq_length = 5
        data = torch.randn(batch_size, seq_length, *dmbd_model.obs_shape, device=torch_device)
        
        # Move model to device
        dmbd_model = dmbd_model.to(torch_device)
        
        # Initial loss
        output = dmbd_model(data)
        initial_loss = dmbd_model.loss(data, output)
        
        # Training loop
        n_steps = 5
        losses = []
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            output = dmbd_model(data)
            loss = dmbd_model.loss(data, output)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Verify loss decreased
        assert losses[-1] < initial_loss, "Loss did not decrease during training"
        
        # Check gradient flow
        for name, param in dmbd_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Infinite gradient for {name}"
        
    except Exception as e:
        pytest.fail(f"Gradient flow test failed: {e}")

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
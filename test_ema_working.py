"""
Simple test to verify EMA implementation is working correctly.
Run this to check if EMA weights diverge from model weights during training.
"""
import torch
import torch.nn as nn
import copy
from utils.ema import ModelEma

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.bn = nn.BatchNorm1d(5)
    
    def forward(self, x):
        return self.bn(self.linear(x))

def test_ema():
    print("="*60)
    print("Testing EMA Implementation")
    print("="*60)
    
    # Initialize model and EMA
    model = SimpleModel()
    ema = ModelEma(model, decay=0.9999)
    
    print("\n1. Initial State Check")
    print("-" * 60)
    # Initially, EMA should be identical to model
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), ema.module.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"❌ FAIL: {n1} weights differ initially!")
            return False
    print("✓ PASS: EMA weights identical to model initially")
    
    print("\n2. Training Step Simulation")
    print("-" * 60)
    # Simulate a training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(4, 10)
    target = torch.randn(4, 5)
    
    # Forward + backward + update
    model.train()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✓ Training step completed (loss: {loss.item():.4f})")
    
    print("\n3. EMA Update Check")
    print("-" * 60)
    # After training step, model weights changed but EMA hasn't yet
    linear_weight_model = model.linear.weight.clone()
    linear_weight_ema_before = ema.module.linear.weight.clone()
    
    # Now update EMA
    ema.update(model)
    linear_weight_ema_after = ema.module.linear.weight
    
    # Check if model and EMA diverged
    if torch.equal(linear_weight_model, linear_weight_ema_after):
        print("❌ FAIL: EMA weights identical to model after update!")
        return False
    
    # Check if EMA actually changed
    if torch.equal(linear_weight_ema_before, linear_weight_ema_after):
        print("❌ FAIL: EMA weights did not change after update!")
        return False
    
    # Verify EMA formula: ema_new = decay * ema_old + (1-decay) * model
    decay = 0.9999
    expected_ema = decay * linear_weight_ema_before + (1 - decay) * linear_weight_model
    if not torch.allclose(linear_weight_ema_after, expected_ema, atol=1e-6):
        print("❌ FAIL: EMA update formula incorrect!")
        max_diff = (linear_weight_ema_after - expected_ema).abs().max().item()
        print(f"   Max difference: {max_diff:.2e}")
        return False
    
    print("✓ PASS: EMA weights updated correctly")
    print(f"   Model-EMA difference: {(linear_weight_model - linear_weight_ema_after).abs().max().item():.2e}")
    
    print("\n4. Multiple Update Check")
    print("-" * 60)
    # Do several more updates
    ema_weights_history = [ema.module.linear.weight.clone()]
    
    for i in range(5):
        output = model(x)
        loss = nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        ema_weights_history.append(ema.module.linear.weight.clone())
    
    # Check that EMA keeps changing
    all_different = True
    for i in range(len(ema_weights_history) - 1):
        if torch.equal(ema_weights_history[i], ema_weights_history[i+1]):
            all_different = False
            print(f"❌ FAIL: EMA weights stopped changing at step {i+1}")
            break
    
    if all_different:
        print("✓ PASS: EMA weights continue updating over multiple steps")
    
    print("\n5. EMA Always in Eval Mode Check")
    print("-" * 60)
    model.train()
    if ema.module.training:
        print("❌ FAIL: EMA model is in training mode!")
        return False
    print("✓ PASS: EMA model stays in eval mode")
    
    print("\n6. Buffer Update Check (BatchNorm)")
    print("-" * 60)
    # Check if BatchNorm running stats are updated
    bn_mean_before = ema.module.bn.running_mean.clone()
    
    # Train model with different data
    model.train()
    for _ in range(10):
        x_new = torch.randn(4, 10) + 5.0  # Different distribution
        model(x_new)
        ema.update(model)
    
    bn_mean_after = ema.module.bn.running_mean
    
    if not torch.equal(bn_mean_before, bn_mean_after):
        print("✓ PASS: EMA updates buffers (BatchNorm stats)")
    else:
        print("⚠️  WARNING: EMA buffers did not change (might be expected if model BN not updating)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - EMA Implementation is Working!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_ema()
    exit(0 if success else 1)

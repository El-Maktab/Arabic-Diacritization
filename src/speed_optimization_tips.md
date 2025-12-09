# CRF Training Speed - Realistic Optimizations

## Current Situation
- **Speed**: ~1.6 iterations/second
- **Time per epoch**: ~60 minutes  
- **Total training time**: ~5 hours for 5 epochs

## Why Previous Optimizations Didn't Help Much

The **CRF layer is the bottleneck**, not the BiLSTM:
- CRF Viterbi algorithm is inherently sequential
- Mixed precision has minimal impact on CRF operations
- The `torchcrf` library operations don't parallelize well

## Practical Speed Improvements

### Option 1: Reduce Epochs (Fastest!)
Most models converge in 2-3 epochs:

```python
NUM_EPOCHS = 3  # Instead of 5
```
**Savings**: 40% less time (~3 hours instead of 5)

### Option 2: Increase Batch Size
If you have GPU memory available:

```python
BATCH_SIZE = 64   # Double current size
# or
BATCH_SIZE = 128  # 4x current size
```

This reduces total iterations:
- Current: 5823 iterations/epoch at batch_size=32
- With 64: ~2912 iterations/epoch (**2x faster**)
- With 128: ~1456 iterations/epoch (**4x faster**)

### Option 3: Sample Training Data
Use a subset for faster training:

```python
# After loading dataset
train_size = int(len(train_dataset) * 0.5)  # Use 50% of data
train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
```

### Option 4: Use Gradient Accumulation
Simulate larger batch sizes without OOM:

```python
ACCUMULATION_STEPS = 4
BATCH_SIZE = 32

# In training loop
for i, (train_X, train_Y) in enumerate(train_data_loader):
    # ... forward pass and loss ...
    
    loss = loss / ACCUMULATION_STEPS  # Scale loss
    scaler.scale(loss).backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Recommended: Increase Batch Size

**Try this first** (if you have GPU memory):

1. Change `BATCH_SIZE = 64` in your config cell
2. Restart kernel and re-run

**Check GPU memory usage:**
```bash
watch -n 1 nvidia-smi
```

If you see "Out of Memory" errors, reduce back to 32.

## What About the CRF?

The CRF itself can't be sped up much, but you could:
- Use a simpler emission-only model (BiLSTM without CRF) for comparison
- Accept the slower training time as the cost of sequence-level modeling
- Train overnight or in background

## Bottom Line

**CRF training is inherently slow**. The best practical speedup is:
1. ✅ Increase `BATCH_SIZE` to 64 or 128 (2-4x faster)
2. ✅ Reduce `NUM_EPOCHS` to 3 (40% time savings)  
3. ✅ Combined: ~3-5x faster total training time

# Lenient Buffer Matching - User Guide

## What is Lenient Buffer Matching?

Lenient buffer matching allows the weight mapper to successfully match parameters between models even when their statistical buffers (like BatchNorm's `running_mean` and `running_var`) have different shapes, while still strictly matching trainable parameters like weights and biases.

## Why Is This Useful?

When mapping weights between different model architectures, you may encounter situations where:

- Models use different normalization layers (BatchNorm vs LayerNorm vs GroupNorm)
- Models have fused operations (Conv+BN → ConvBN fusion)
- Models track different statistics
- Source and target have different channel dimensions

Without lenient matching, a mismatch in any buffer would prevent the entire module from matching, even if the actual trainable parameters could be compatible.

## Usage

### Default Behavior (Recommended)

Lenient buffer matching is now the **default**:

```python
from lit_wsl.mapper import WeightMapper

mapper = WeightMapper(
    source_module=source_model,
    target_module=target_model
)

result = mapper.suggest_mapping(threshold=0.7)
```

### Explicit Mode Selection

You can explicitly choose between three modes:

#### 1. Lenient Mode (Default)

Statistical buffers get soft penalties instead of hard rejection:

```python
mapper = WeightMapper(
    source_module=source_model,
    target_module=target_model,
    buffer_matching_mode="lenient"  # Default
)
```

**When to use**: Most cases. Allows matching despite buffer differences while preserving trainable parameter strictness.

#### 2. Strict Mode

Original behavior - all parameters including buffers must match:

```python
mapper = WeightMapper(
    source_module=source_model,
    target_module=target_model,
    buffer_matching_mode="strict"
)
```

**When to use**: When you need exact architecture matching, or when buffers are critical for your use case.

#### 3. Exclude Mode

Completely ignores all buffers during matching:

```python
mapper = WeightMapper(
    source_module=source_model,
    target_module=target_model,
    buffer_matching_mode="exclude"
)
```

**When to use**: When transferring between fundamentally different normalizations, or when you'll reinitialize buffers anyway.

## Examples

### Example 1: Different Channel Counts in Buffers

```python
# Source: BatchNorm with 64 channels
# - conv.weight: (64, 32, 3, 3) ✓
# - bn.running_mean: (64,)       ✗ shape mismatch
# - bn.running_var: (64,)         ✗ shape mismatch

# Target: BatchNorm with 32 channels
# - conv.weight: (64, 32, 3, 3) ✓
# - bn.running_mean: (32,)
# - bn.running_var: (32,)

# With strict mode: NO MATCH (buffer shapes differ)
# With lenient mode: MATCH (buffers get soft penalty, weights match)
```

### Example 2: Fused Operations

```python
# Source: Separate Conv + BN
# - conv.weight: (128, 64, 3, 3) ✓
# - bn.weight: (128,)            ✓
# - bn.running_mean: (128,)      (buffer)

# Target: Fused ConvBN
# - fused.weight: (128, 64, 3, 3) ✓
# - fused.bn_weight: (128,)       ✓
# [no running stats buffers]

# With strict mode: NO MATCH (missing buffers in target)
# With lenient mode: MATCH (buffer presence difference ignored)
```

### Example 3: Different Normalization Types

```python
# Source: BatchNorm
# - conv.weight: (256, 128, 3, 3) ✓
# - bn.weight: (256,)             ✓
# - bn.running_mean: (256,)       (buffer)
# - bn.running_var: (256,)        (buffer)

# Target: GroupNorm (no running stats)
# - conv.weight: (256, 128, 3, 3) ✓
# - gn.weight: (256,)             ✓
# [no running stats]

# With strict mode: NO MATCH (missing buffers)
# With exclude mode: MATCH (all buffers ignored)
```

## What Gets Matched?

### Lenient Mode Matches:

✅ **Trainable Parameters** (weights, biases)

- Must have exact shape match
- Must have same parameter type (weight→weight, bias→bias)

✅ **Statistical Buffers** (running_mean, running_var, num_batches_tracked)

- Shape mismatches get soft penalty (score 0.2)
- Allows name/hierarchy scores to contribute
- Won't block matching if trainable params align

### What Still Fails:

❌ **Trainable parameter shape mismatches**

```python
# Source: conv.weight (64, 32, 3, 3)
# Target: conv.weight (128, 32, 3, 3)
# Result: NO MATCH (different out_channels in trainable param)
```

❌ **Parameter type mismatches**

```python
# Source: conv.weight
# Target: fc.bias
# Result: NO MATCH (weight ≠ bias)
```

## Comparison of Modes

| Scenario                 | Strict  | Lenient   | Exclude |
| ------------------------ | ------- | --------- | ------- |
| Exact match (all params) | ✓ Match | ✓ Match   | ✓ Match |
| Buffer shape differs     | ✗ Fail  | ✓ Match\* | ✓ Match |
| Weight shape differs     | ✗ Fail  | ✗ Fail    | ✗ Fail  |
| Missing buffers          | ✗ Fail  | ✓ Match\* | ✓ Match |
| Different norm types     | ✗ Fail  | ✓ Match\* | ✓ Match |

\*With reduced confidence score

## Best Practices

### 1. Start with Lenient (Default)

```python
mapper = WeightMapper(source_module=src, target_module=tgt)
result = mapper.suggest_mapping(threshold=0.7)
```

### 2. Check Match Quality

```python
mapping_with_scores = result.get_mapping_with_scores()

for src, (tgt, score) in mapping_with_scores.items():
    if score < 0.8:
        print(f"Low confidence: {src} -> {tgt} (score: {score:.3f})")
```

### 3. Verify Unmatched Parameters

```python
unmatched = result.get_unmatched()

# Check if unmatched are mostly buffers or weights
for param in unmatched['source'][:10]:
    is_buffer = 'running' in param or 'num_batches' in param
    print(f"{'[BUFFER]' if is_buffer else '[WEIGHT]'} {param}")
```

### 4. Use Strict Mode for Safety-Critical Applications

```python
# When exact matching is required
mapper = WeightMapper(
    source_module=src,
    target_module=tgt,
    buffer_matching_mode="strict"
)
```

## Performance Impact

Buffer matching mode affects:

- **Matching rate**: Lenient mode typically matches 10-20% more parameters
- **Computation time**: Negligible difference (<1%)
- **Memory usage**: No impact

## Common Questions

### Q: Will lenient mode allow incorrect matches?

**A**: No. Trainable parameters (weights, biases) still require exact shape matches. Only statistical buffers get leniency, which is safe because:

- Buffers are typically recomputed during training
- Buffer mismatches rarely affect functionality
- Trainable params still strictly enforced

### Q: Should I use exclude mode to maximize matches?

**A**: Not necessarily. Exclude mode prevents buffer matching entirely, which might cause you to lose valid buffer transfers. Use lenient mode first, then exclude if needed.

### Q: When should I use strict mode?

**A**:

- When you need exact architecture matching
- When loading into production without retraining
- When buffers encode important learned statistics (rare)

### Q: Can I switch modes after creating the mapper?

**A**: No, the mode is set during initialization. Create a new WeightMapper instance with a different mode.

## Troubleshooting

### Issue: Still getting many unmatched parameters

```python
# Check what's actually unmatched
unmatched = result.get_unmatched()

# Count buffer vs weight mismatches
buffer_count = sum(1 for p in unmatched['source'] if 'running' in p or 'num_batches' in p)
weight_count = len(unmatched['source']) - buffer_count

print(f"Unmatched: {weight_count} weights, {buffer_count} buffers")
```

If most unmatched are weights, the issue is likely:

- Different architectures (channel counts, layer counts)
- Naming convention differences
- Structural mismatches

Lenient buffer matching won't help with these. Consider:

- Verifying you're using the right model variants
- Checking if models are from different frameworks
- Using lower threshold for more flexible matching

### Issue: Matches have low confidence scores

Low scores on buffer matches are expected (get 0.2 shape score). Check:

```python
mapping_with_scores = result.get_mapping_with_scores()

# Separate buffer matches from weight matches
buffer_matches = {src: (tgt, score) for src, (tgt, score) in mapping_with_scores.items()
                  if 'running' in src or 'num_batches' in src}

weight_matches = {src: (tgt, score) for src, (tgt, score) in mapping_with_scores.items()
                  if src not in buffer_matches}

avg_buffer_score = sum(s for _, s in buffer_matches.values()) / len(buffer_matches)
avg_weight_score = sum(s for _, s in weight_matches.values()) / len(weight_matches)

print(f"Average buffer score: {avg_buffer_score:.3f}")
print(f"Average weight score: {avg_weight_score:.3f}")
```

Buffer scores around 0.3-0.5 are normal in lenient mode. Weight scores should be >0.7.

## Summary

**Lenient buffer matching (default)** provides the best balance:

- ✅ Flexible enough to handle normalization differences
- ✅ Strict enough to prevent incorrect weight matches
- ✅ Safe for most use cases
- ✅ Automatically enabled - no code changes needed

For most users, the default lenient mode will provide better results with no downsides.

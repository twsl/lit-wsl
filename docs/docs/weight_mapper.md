# WeightMapper Documentation

## Overview

The `WeightMapper` class is a powerful tool for analyzing PyTorch models and automatically suggesting weight mapping dictionaries. This is particularly useful when you need to transfer weights from one model architecture to a slightly different architecture (e.g., after refactoring layer names or reorganizing modules).

**Common Use Cases:**

1. **Model-to-Model**: Compare two model instances when you have both architectures
2. **Checkpoint-to-Model**: Map weights from a saved checkpoint to a new model (most common in production)
3. **Architecture Refactoring**: Adapt weights after renaming or reorganizing layers

## Key Features

- **Intelligent Matching**: Uses a multi-factor scoring system based on:
  - Exact tensor shape matching (required)
  - Name similarity (edit distance, token overlap, common substrings)
  - Hierarchical position similarity
- **Multiple Strategies**:
  - `best_match`: Finds highest scoring match for each source parameter
  - `conservative`: Only suggests high-confidence matches (threshold ≥ 0.8)
  - `shape_only`: Matches based purely on tensor shapes
- **Comprehensive Analysis**:
  - Coverage statistics
  - Confidence scores for each mapping
  - Unmatched parameter detection
  - JSON report export

## How It Works

### 1. Parameter Extraction

The mapper extracts all parameters from both models and builds metadata including:

- Parameter name and hierarchical path
- Tensor shape, dtype, and element count
- Depth in module hierarchy
- Name tokens for similarity matching

### 2. Scoring Algorithm

For each pair of source and target parameters with matching shapes, a composite score is calculated:

```python
score = 0.5 * shape_score + 0.3 * name_score + 0.2 * hierarchy_score
```

Where:

- **Shape Score** (0.0-1.0):

  - 1.0 for exact match
  - 0.7 for transposed shapes (e.g., different Conv implementations)
  - 0.0 for different shapes

- **Name Score** (0.0-1.0):

  - Token overlap (Jaccard similarity)
  - Edit distance (Levenshtein)
  - Longest common substring
  - Parameter type match (weight, bias, etc.)

- **Hierarchy Score** (0.0-1.0):
  - Depth similarity
  - Module path matching
  - Common parent modules

### 3. Mapping Generation

The algorithm:

1. Filters candidates by exact shape match
2. Computes composite scores for all candidates
3. Selects the highest scoring match for each source parameter
4. Ensures one-to-one mappings (no duplicate target assignments)
5. Applies threshold filtering

## Usage Examples

### Option 1: From Checkpoint (Recommended)

This is the most common use case - you have a checkpoint file but not the original model code:

```python
from lit_wsl.models.weight_mapper import WeightMapper
from lit_wsl.models.weight_renamer import WeightRenamer
from lit_wsl.models.checkpoint import load_checkpoint_as_dict

# Load checkpoint
checkpoint = load_checkpoint_as_dict("old_model.pth")
if "state_dict" in checkpoint:
    old_weights = checkpoint["state_dict"]
else:
    old_weights = checkpoint

# Create new model
new_model = NewModel()

# Create mapper from checkpoint
mapper = WeightMapper.from_state_dict(old_weights, new_model)

# Generate and review mapping
mapping = mapper.suggest_mapping(threshold=0.6)
mapper.print_analysis()

# Apply mapping to checkpoint
renamer = WeightRenamer("old_model.pth")
renamer.rename_keys(mapping)
renamer.save("adapted_weights.pth")

# Load into new model
new_model.load_state_dict(torch.load("adapted_weights.pth"))
```

### Option 2: From Two Models

When you have access to both model architectures:

```python
from lit_wsl.models.weight_mapper import WeightMapper

# Create mapper with two models
old_model = OldModel()
new_model = NewModel()

mapper = WeightMapper(old_model, new_model)

# Generate mapping
mapping = mapper.suggest_mapping(threshold=0.6)

# Print analysis
mapper.print_analysis()

# Get the mapping dictionary
mapping_dict = mapper.get_mapping_dict()
```

### Option 3: Directly from Checkpoint File

Convenience method that loads and processes the checkpoint:

```python
from lit_wsl.models.weight_mapper import WeightMapper

# Create mapper directly from checkpoint file
new_model = NewModel()
mapper = WeightMapper.from_checkpoint("old_model.pth", new_model)

# Generate mapping
mapping = mapper.suggest_mapping()
```

### Basic Usage

```python
from lit_wsl.models.weight_mapper import WeightMapper

# Create mapper with two models
mapper = WeightMapper(old_model, new_model)

# Generate mapping
mapping = mapper.suggest_mapping(threshold=0.6)

# Print analysis
mapper.print_analysis()

# Get the mapping dictionary
mapping_dict = mapper.get_mapping_dict()
```

### Using with WeightRenamer

```python
from lit_wsl.models.weight_mapper import WeightMapper
from lit_wsl.models.weight_renamer import WeightRenamer

# 1. Analyze models and get mapping
mapper = WeightMapper(old_model, new_model)
mapping = mapper.suggest_mapping()
mapper.print_analysis()

# 2. Apply mapping to checkpoint
renamer = WeightRenamer("old_weights.pth")
renamer.rename_keys(mapping)
renamer.save("adapted_weights.pth")

# 3. Load adapted weights into new model
new_model.load_state_dict(torch.load("adapted_weights.pth"))
```

### Conservative Matching

For critical applications where you want high confidence:

```python
# Only suggest matches with score ≥ 0.8
mapping = mapper.suggest_mapping(strategy="conservative")
```

### Shape-Only Matching

When names have completely changed but architecture is the same:

```python
# Match purely by tensor shapes
mapping = mapper.suggest_mapping(strategy="shape_only")
```

### Custom Weights

Adjust the scoring weights for your use case:

```python
# Prioritize name similarity over hierarchy
custom_weights = {"shape": 0.5, "name": 0.4, "hierarchy": 0.1}
mapping = mapper.suggest_mapping(weights=custom_weights)
```

### Export Report

Generate a detailed JSON report for review:

```python
mapper.export_mapping_report("mapping_report.json")
```

The report includes:

- Model names and parameter counts
- All mappings with confidence scores and shapes
- Unmatched parameters from both models
- Coverage statistics

### Get Mappings with Scores

```python
# Get list of (source, target, score) tuples
mappings = mapper.get_mapping_with_scores()

# Sort by confidence
mappings.sort(key=lambda x: x[2], reverse=True)

# Review low-confidence matches
low_confidence = [(s, t, score) for s, t, score in mappings if score < 0.7]
```

## API Reference

### WeightMapper Class

#### Constructors

##### `__init__()`

```python
WeightMapper(
    source_module: nn.Module | None = None,
    target_module: nn.Module | None = None,
    shape_tolerance: float = 0.0
)
```

**Parameters:**

- `source_module`: The source model (with weights to adapt from)
- `target_module`: The target model (to adapt weights to)
- `shape_tolerance`: Relative tolerance for shape matching (default: 0.0 = exact match only)

##### `from_state_dict()` (Recommended)

```python
@classmethod
WeightMapper.from_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_module: nn.Module,
    shape_tolerance: float = 0.0
) -> WeightMapper
```

Create a WeightMapper from a source state dictionary and target module.

**Parameters:**

- `source_state_dict`: State dictionary from the source model (e.g., loaded checkpoint)
- `target_module`: The target model to adapt weights to
- `shape_tolerance`: Relative tolerance for shape matching (default: 0.0)

**Returns:** WeightMapper instance

**Example:**

```python
from lit_wsl.models.weight_mapper import WeightMapper
from lit_wsl.models.checkpoint import load_checkpoint_as_dict

# Load old weights
checkpoint = load_checkpoint_as_dict("old_model.pth")
if "state_dict" in checkpoint:
    old_weights = checkpoint["state_dict"]
else:
    old_weights = checkpoint

# Create mapper with new model
new_model = NewModel()
mapper = WeightMapper.from_state_dict(old_weights, new_model)
mapping = mapper.suggest_mapping()
```

##### `from_checkpoint()`

```python
@classmethod
WeightMapper.from_checkpoint(
    checkpoint_path: str | Path,
    target_module: nn.Module,
    shape_tolerance: float = 0.0
) -> WeightMapper
```

Create a WeightMapper from a checkpoint file and target module. Automatically handles nested checkpoint structures (e.g., Lightning checkpoints with "state_dict" key).

**Parameters:**

- `checkpoint_path`: Path to the checkpoint file
- `target_module`: The target model to adapt weights to
- `shape_tolerance`: Relative tolerance for shape matching (default: 0.0)

**Returns:** WeightMapper instance

**Example:**

```python
from lit_wsl.models.weight_mapper import WeightMapper

new_model = NewModel()
mapper = WeightMapper.from_checkpoint("old_model.pth", new_model)
mapping = mapper.suggest_mapping()
```

#### Methods

##### suggest_mapping()

```python
suggest_mapping(
    threshold: float = 0.6,
    strategy: str = "best_match",
    weights: dict[str, float] | None = None
) -> dict[str, str]
```

Generate parameter name mapping.

**Parameters:**

- `threshold`: Minimum score threshold (0.0-1.0)
- `strategy`: `"best_match"`, `"conservative"`, or `"shape_only"`
- `weights`: Custom scoring weights dict with keys: `"shape"`, `"name"`, `"hierarchy"`

**Returns:** Dictionary mapping source parameter names to target parameter names

##### print_analysis()

```python
print_analysis(
    top_n: int = 10,
    show_unmatched: bool = True
) -> None
```

Print detailed analysis of the mapping.

**Parameters:**

- `top_n`: Number of top mappings to display
- `show_unmatched`: Whether to show unmatched parameters

##### get_unmatched()

```python
get_unmatched() -> dict[str, list[str]]
```

**Returns:** Dictionary with `"source"` and `"target"` keys containing lists of unmatched parameter names

##### get_mapping_dict()

```python
get_mapping_dict() -> dict[str, str]
```

**Returns:** Copy of the current mapping dictionary

##### get_mapping_with_scores()

```python
get_mapping_with_scores() -> list[tuple[str, str, float]]
```

**Returns:** List of `(source_name, target_name, score)` tuples

##### export_mapping_report()

```python
export_mapping_report(output_path: str | Path) -> None
```

Export detailed mapping report to a JSON file.

## Best Practices

1. **Review Results**: Always call `print_analysis()` to review the suggested mappings before using them.

2. **Check Coverage**: Low coverage might indicate significant architectural differences. Investigate unmatched parameters.

3. **Validate Shapes**: The mapper requires exact shape matches by default. This is a safety feature.

4. **Start Conservative**: Begin with a higher threshold and gradually lower it if needed.

5. **Manual Override**: You can edit the mapping dictionary before applying it:

   ```python
   mapping = mapper.get_mapping_dict()
   # Manually adjust specific mappings
   mapping["old.layer"] = "new.corrected_layer"
   ```

6. **Test Loading**: After adapting weights, test that the model loads correctly:
   ```python
   try:
       new_model.load_state_dict(adapted_weights)
       print("✓ Weights loaded successfully")
   except Exception as e:
       print(f"✗ Error loading weights: {e}")
   ```

## Limitations

- Requires exact shape matching (by default)
- Cannot handle structural changes (e.g., layer merging/splitting)
- Name-based scoring may fail for completely unrelated naming schemes
- One-to-one mapping only (no parameter merging or splitting)

## Example Output

```
================================================================================
Weight Mapping Analysis
================================================================================

Source model: OldModel
  Total parameters: 8

Target model: NewModel
  Total parameters: 8

Matching results:
  Matched: 8
  Coverage: 100.0%

----------------------------Top suggested mappings:------------------------------
Source                                   → Target                        Score
--------------------------------------------------------------------------------
classifier.0.weight                      → head.0.weight                 0.780
classifier.2.weight                      → head.2.weight                 0.780
classifier.0.bias                        → head.0.bias                   0.773
backbone.0.weight                        → feature_extractor.0.weight    0.759
  ... and 4 more matches
================================================================================
```

## See Also

- [WeightRenamer](./weight_renamer.md) - For applying the mapping to checkpoint files
- [test_weight_mapper.py](../scripts/test_weight_mapper.py) - Complete examples and demonstrations

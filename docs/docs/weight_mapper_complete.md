# WeightMapper: Complete Guide

This comprehensive guide covers both the usage and architecture of the WeightMapper, a powerful tool for analyzing PyTorch models and automatically mapping weights between different architectures.

## Table of Contents

### Part 1: User Guide

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Usage Examples](#usage-examples)
4. [API Reference](#api-reference)
5. [Best Practices](#best-practices)

### Part 2: Architecture & Implementation

6. [Group-Based Parameter Mapping](#group-based-parameter-mapping)
7. [Hierarchical Structure Extraction](#hierarchical-structure-extraction)
8. [Implementation Details](#implementation-details)
9. [Testing and Validation](#testing-and-validation)
10. [Future Enhancements](#future-enhancements)

---

# Part 1: User Guide

## Overview

The `WeightMapper` class is a powerful tool for analyzing PyTorch models and automatically suggesting weight mapping dictionaries. This is particularly useful when you need to transfer weights from one model architecture to a slightly different architecture (e.g., after refactoring layer names or reorganizing modules).

**Common Use Cases:**

1. **Model-to-Model**: Compare two model instances when you have both architectures
2. **Checkpoint-to-Model**: Map weights from a saved checkpoint to a new model (most common in production)
3. **Architecture Refactoring**: Adapt weights after renaming or reorganizing layers

## Key Features

- **Intelligent Matching**: Uses a multi-factor scoring system based on:
  - Exact tensor shape matching (required)
  - Parameter type matching (weight→weight, bias→bias, etc.)
  - Name similarity (edit distance, token overlap, common substrings)
  - Hierarchical position similarity
  - Execution order similarity (optional, when dummy_input provided)
- **Multiple Strategies**:
  - `best_match`: Finds highest scoring match for each source parameter
  - `conservative`: Only suggests high-confidence matches (threshold ≥ 0.75)
  - `shape_only`: Matches based purely on tensor shapes
- **Comprehensive Analysis**:
  - Coverage statistics
  - Confidence scores for each mapping
  - Unmatched parameter detection
  - JSON report export
- **Optional Performance Enhancement**:
  - Execution order tracking via `dummy_input` parameter
  - Provides ~2% average score improvement
  - Particularly helpful for models with significant structural changes

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
  - Execution order similarity (when dummy_input provided)

**Note:** When `dummy_input` is provided, the mapper runs a forward pass to track layer execution order, adding an execution order component to the hierarchy score. This provides measurable improvements (~2% on average) but is entirely optional.

### 3. Mapping Generation

The algorithm:

1. Filters candidates by exact shape match
2. Computes composite scores for all candidates
3. Selects the highest scoring match for each source parameter
4. Ensures one-to-one mappings (no duplicate target assignments)
5. Applies threshold filtering

## Usage Examples

### Basic Usage (Recommended Starting Point)

The simplest way to use WeightMapper - no dummy input needed:

```python
from lit_wsl.models.weight_mapper import WeightMapper
from lit_wsl.models.weight_renamer import WeightRenamer

# Create mapper from checkpoint
new_model = NewModel()
mapper = WeightMapper.from_checkpoint("old_model.pth", new_model)

# Generate and review mapping
mapping = mapper.suggest_mapping(threshold=0.6)
mapper.print_analysis()

# Apply mapping to checkpoint
renamer = WeightRenamer("old_model.pth")
renamer.rename_keys(mapping)
renamer.save("adapted_weights.pth")

# Load adapted weights
new_model.load_state_dict(torch.load("adapted_weights.pth"))
```

### With Execution Order Tracking (Optional Enhancement)

For better matching scores (~2% improvement on average), provide a dummy input:

```python
import torch
from lit_wsl.models.weight_mapper import WeightMapper

# Create dummy input matching your model's expected input
dummy_input = torch.randn(1, 3, 224, 224)  # [batch, channels, height, width]

# Create mapper with execution order tracking
new_model = NewModel()
mapper = WeightMapper.from_checkpoint(
    "old_model.pth",
    new_model,
    dummy_input=dummy_input  # Optional but improves matching
)

mapping = mapper.suggest_mapping(threshold=0.6)
```

### Option 1: From Checkpoint (Most Common)

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

## Execution Order Tracking (Optional)

### What is it?

When you provide a `dummy_input` tensor, the WeightMapper performs a forward pass through both models to track the order in which layers are executed. This execution order is then incorporated into the similarity scoring.

### When to use it?

- **Recommended for:** Complex models with significant structural changes
- **Optional for:** Simple models or models with similar structures
- **Average improvement:** ~2% score increase (83% of parameters improved in tests)
- **No downside:** Never hurts matching quality, only improves or maintains it

### Usage

```python
import torch
from lit_wsl.models.weight_mapper import WeightMapper

# Define dummy input matching your model's expected input shape
dummy_input = torch.randn(1, 3, 224, 224)  # [batch, channels, height, width]

# Option 1: With both models
mapper = WeightMapper(old_model, new_model, dummy_input=dummy_input)

# Option 2: From checkpoint
mapper = WeightMapper.from_checkpoint("old.pth", new_model, dummy_input=dummy_input)

# Option 3: From state dict
mapper = WeightMapper.from_state_dict(state_dict, new_model, dummy_input=dummy_input)

mapping = mapper.suggest_mapping()
```

### Performance Impact

Based on test results with models that have renamed layers:

```
Without execution order:
  Average score: 0.7529

With execution order:
  Average score: 0.7695
  Improvement: +0.0167 (+2.21%)

Individual parameter changes:
  Improved:   83.3% of parameters
  Unchanged:  16.7% of parameters
  Worse:      0.0% of parameters
```

### Tips

- Use the same input shape your model expects during inference
- Batch size of 1 is sufficient for order tracking
- The forward pass runs once during initialization, not during `suggest_mapping()`
- If the forward pass fails, the mapper gracefully falls back to working without execution order

## API Reference

### WeightMapper Class

#### Constructors

##### `__init__()`

```python
WeightMapper(
    source_module: nn.Module | None = None,
    target_module: nn.Module | None = None,
    shape_tolerance: float = 0.0,
    dummy_input: torch.Tensor | None = None,
    incompatible_pairs: list[tuple[set[str], set[str]]] | None = None
)
```

**Parameters:**

- `source_module`: The source model (with weights to adapt from)
- `target_module`: The target model (to adapt weights to)
- `shape_tolerance`: Relative tolerance for shape matching (default: 0.0 = exact match only)
- `dummy_input`: (Optional) Dummy input tensor for execution order tracking. Improves matching by ~2% on average. Not required.
- `incompatible_pairs`: (Optional) List of incompatible component pairs for semantic matching. Each pair is a tuple of two sets of semantic chunks. For example: `[({{"backbone"}}, {{"head"}})]` prevents backbone modules from matching with head modules. If `None`, uses default pairs that prevent common cross-component matches (e.g., backbone vs head, encoder vs decoder). Pass an empty list `[]` to disable all incompatibility checks.

##### `from_state_dict()` (Recommended)

```python
@classmethod
WeightMapper.from_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_module: nn.Module,
    shape_tolerance: float = 0.0,
    dummy_input: torch.Tensor | None = None,
    incompatible_pairs: list[tuple[set[str], set[str]]] | None = None
) -> WeightMapper
```

Create a WeightMapper from a source state dictionary and target module.

**Parameters:**

- `source_state_dict`: State dictionary from the source model (e.g., loaded checkpoint)
- `target_module`: The target model to adapt weights to
- `shape_tolerance`: Relative tolerance for shape matching (default: 0.0)
- `dummy_input`: (Optional) Dummy input tensor for execution order tracking
- `incompatible_pairs`: (Optional) List of incompatible component pairs. If None, no restrictions. See `__init__()` for details.

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

## Customizing Incompatible Component Pairs

The WeightMapper allows you to optionally define incompatible component pairs to prevent clearly incorrect cross-component matches (e.g., preventing backbone modules from matching with head modules).

### Default Behavior

By default, the mapper applies **no cross-component restrictions**:

```python
mapper = WeightMapper(source_model, target_model)
# Allows: backbone.conv1 → head.conv1 (if shapes and names match)
# Allows: encoder.layer1 → decoder.layer1 (if shapes and names match)
# All matches are evaluated purely on shape, name similarity, and hierarchy
```

### Adding Incompatibility Restrictions

Define your own incompatible pairs for domain-specific architectures:

```python
# Example: NLP model with custom components
custom_pairs = [
    ({"encoder", "embedding"}, {"decoder", "output"}),
    ({"attention"}, {"feedforward", "mlp"}),
]

mapper = WeightMapper(
    source_model,
    target_model,
    incompatible_pairs=custom_pairs
)
```

**How it works:**

- Each pair is a tuple of two sets of semantic chunks
- Module names are split into chunks (e.g., "yolo_backbone" → {"yolo", "backbone"})
- If a module contains chunks from one set and another module contains chunks from the other set, they won't match
- Checks are symmetric (A→B == B→A)

### Example Use Cases

**Vision Models (Prevent Backbone-Head Confusion):**

```python
# Explicitly prevent backbone from matching with head components
vision_pairs = [
    ({"backbone", "encoder", "feature"}, {"head", "classifier", "decoder"}),
    ({"backbone", "encoder"}, {"neck", "fpn"}),
]
mapper = WeightMapper(resnet_model, vit_model, incompatible_pairs=vision_pairs)
```

**Vision Transformers:**

```python
vit_pairs = [
    ({"patch", "embedding"}, {"head", "classifier"}),
    ({"encoder", "transformer"}, {"decoder", "mlp_head"}),
]
mapper = WeightMapper(vit_model, target_model, incompatible_pairs=vit_pairs)
```

**Detection Models:**

```python
detection_pairs = [
    ({"backbone", "resnet", "vgg"}, {"rpn", "head", "neck"}),
    ({"fpn", "pafpn"}, {"roi", "head"}),
]
mapper = WeightMapper(yolo_v5, yolo_v8, incompatible_pairs=detection_pairs)
```

**Default (No Restrictions):**

```python
# For models with custom naming or when you want maximum flexibility
mapper = WeightMapper(
    old_model,
    new_model
    # incompatible_pairs not specified - no restrictions applied
)
```

## Best Practices

1. **Review Results**: Always call `print_analysis()` or `visualize_mapping()` to review the suggested mappings before using them.

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

## Visualization Features

The WeightMapper includes powerful Rich-based visualization tools for exploring model hierarchies and analyzing mapping results with beautiful, color-coded output.

### Visualizing Mapping Results

Display mapping results with color-coded confidence scores, transformation indicators, and statistics:

```python
from lit_wsl.mapper.weight_mapper import WeightMapper

# Create mapper and suggest mapping
mapper = WeightMapper.from_state_dict(checkpoint, new_model)
result = mapper.suggest_mapping(threshold=0.6)

# Visualize with Rich formatting
mapper.visualize_mapping(
    result=result,
    show_unmatched=True,
    max_matches=30,  # Show top 30 matches
    max_unmatched=15  # Show up to 15 unmatched items
)
```

Output includes:

- **Summary Panel**: Coverage, match counts, threshold info
- **Mapping Table**: Source → Target mappings with:
  - Color-coded scores (green >0.8, yellow 0.6-0.8, red <0.6)
  - Transformation indicators (✓ exact, 🔄 transpose, 📐 reshape)
  - Match type badges (GROUP or INDIV)
  - Parameter shapes
- **Unmatched Parameters**: Lists of unmatched source and target parameters

### Visualizing Model Hierarchies

Explore the hierarchical structure of models with side-by-side tree views:

```python
# Show both model hierarchies side by side
mapper.visualize_hierarchies(max_depth=5)

# Highlight matched modules after mapping
mapper.suggest_mapping()
mapper.visualize_hierarchies(show_matches=True, max_depth=5)
```

Features:

- **Tree Structure**: Nested view of all modules
- **Parameter Badges**: W (weight), B (bias) indicators
- **Shape Information**: Parameter shapes shown inline
- **Match Highlighting**: Matched modules shown in green with ✓
- **Depth Limiting**: Control tree depth for readability

### Visualizing Score Breakdowns

Inspect detailed scoring for specific parameters:

```python
# Show detailed score breakdown for a parameter
mapper.visualize_score_breakdown("backbone.conv1.weight")
```

Displays:

- Component scores (shape, name, hierarchy)
- Sub-scores (token, edit distance, LCS, depth, path, order)
- Weight contributions
- Visual progress bars

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

---

# Part 2: Architecture & Implementation

## Group-Based Parameter Mapping

### Overview

The WeightMapper uses **group-based parameter mapping** instead of mapping tensors one by one. This ensures that logically connected parameters (e.g., weight and bias for a layer, or weight/bias/running_mean/running_var for batch normalization) are **assigned together as a cohesive unit**.

### ParameterGroup Class

A new class that represents a group of logically connected parameters:

```python
class ParameterGroup:
    """Represents a group of logically connected parameters (e.g., weight+bias for a layer)."""

    def __init__(self, module_path: str, params: dict[str, ParameterInfo]):
        self.module_path = module_path  # e.g., 'layer1.conv'
        self.params = params  # dict mapping param type to ParameterInfo
        self.param_types = set(params.keys())  # {'weight', 'bias', ...}
```

**Key Features:**

- Groups parameters by their module path
- Stores all parameter types for a module (weight, bias, running_mean, etc.)
- Provides compatibility checking to ensure groups have matching structure

### Parameter Grouping Logic

The mapper automatically groups parameters by their module path:

```python
def _extract_parameter_groups(self, params: dict[str, ParameterInfo]) -> dict[str, ParameterGroup]:
    """Extract parameter groups from parameters."""
    # Groups: {'conv1': ParameterGroup(weight, bias),
    #          'bn1': ParameterGroup(weight, bias), ...}
```

**Example:**

- `conv1.weight` and `conv1.bias` → grouped into `ParameterGroup('conv1', {'weight': ..., 'bias': ...})`
- `bn1.weight`, `bn1.bias` → grouped into `ParameterGroup('bn1', {'weight': ..., 'bias': ...})`

### Group-Based Matching

The `suggest_mapping()` method operates in two stages:

1. **Group Matching** - Matches entire parameter groups based on:
   - Compatible parameter types (same set of param names)
   - Compatible shapes for all parameters
   - Similarity scores averaged across all group parameters

2. **Parameter Expansion** - Expands group mappings to individual parameter mappings

```python
def suggest_mapping(self, threshold: float = 0.6, ...) -> dict[str, str]:
    # Step 1: Match groups
    group_mapping, group_scores = self._suggest_group_mapping(threshold, weights)

    # Step 2: Expand to individual parameters
    for source_module, target_module in group_mapping.items():
        source_group = self.source_groups[source_module]
        target_group = self.target_groups[target_module]

        # Map ALL parameters in the group together
        for param_type in source_group.param_types:
            source_param = source_group.params[param_type]
            target_param = target_group.params[param_type]
            mapping[source_param.name] = target_param.name
```

### Group Compatibility Checking

Groups are only matched if they are compatible:

```python
def is_compatible_with(self, other: ParameterGroup) -> bool:
    # Must have the same parameter types
    if self.param_types != other.param_types:
        return False

    # All parameters must have matching shapes
    for param_type in self.param_types:
        if self.params[param_type].shape != other.params[param_type].shape:
            return False

    return True
```

### Benefits of Group-Based Mapping

#### 1. Atomic Group Assignment

All parameters belonging to a module are assigned together as an atomic unit:

- ✓ `conv1.weight` and `conv1.bias` are ALWAYS mapped together
- ✓ `bn1.weight`, `bn1.bias` are ALWAYS mapped together
- ✗ Can't have `conv1.weight` mapped but `conv1.bias` unmapped

#### 2. Better Semantic Matching

- Groups are matched based on collective similarity, not individual parameter similarity
- The score for a group is the average score of all its parameters
- This provides more robust matching for complex modules

#### 3. Guaranteed Consistency

- If a module path is matched, ALL its parameters are matched
- No partial mappings that could lead to inconsistent state
- Validation ensures parameter types match across groups

#### 4. Handles Complex Modules

Works correctly for modules with multiple parameters:

- Linear layers: `weight`, `bias`
- Conv layers: `weight`, `bias`
- BatchNorm layers: `weight`, `bias`, (and buffers like `running_mean`, `running_var` if treated as parameters)

### Example Output

```
Group mappings (5 groups):
  bn1                            -> encoder_norm1                  (score: 0.701)
    Parameters in group: ['bias', 'weight']
      bn1.bias                                      -> encoder_norm1.bias
      bn1.weight                                    -> encoder_norm1.weight

  conv1                          -> encoder_conv1                  (score: 0.784)
    Parameters in group: ['bias', 'weight']
      conv1.bias                                    -> encoder_conv1.bias
      conv1.weight                                  -> encoder_conv1.weight
```

---

## Hierarchical Structure Extraction

### Overview

The WeightMapper has been enhanced to **extract and leverage the nested hierarchical structure** of neural networks for improved parameter mapping. This builds on the group-based mapping to provide even better matching by understanding parent-child relationships in the model architecture.

### ModuleNode Class - Hierarchical Tree Structure

A new `ModuleNode` class represents the hierarchical structure of modules as a tree:

```python
class ModuleNode:
    """Represents a node in the hierarchical module structure."""

    def __init__(self, name: str, full_path: str, depth: int):
        self.name = name                    # e.g., 'conv1'
        self.full_path = full_path          # e.g., 'encoder.layer1.conv1'
        self.depth = depth                  # 0 for root
        self.children: dict[str, ModuleNode] = {}
        self.parent: ModuleNode | None = None
        self.parameter_group: ParameterGroup | None = None
```

**Benefits:**

- Captures parent-child relationships
- Enables traversal of the module structure
- Provides context for matching decisions

### Hierarchy Building

The `_build_hierarchy()` method constructs a tree from parameter groups:

```python
def _build_hierarchy(self, groups: dict[str, ParameterGroup]) -> ModuleNode:
    """Build a hierarchical tree structure from parameter groups."""
    root = ModuleNode("", "", 0)

    # Sort paths by depth to ensure parents are created before children
    sorted_paths = sorted(groups.keys(), key=lambda x: (x.count('.'), x))

    for module_path in sorted_paths:
        # Create intermediate nodes and attach parameter groups
        ...
```

**Example Structure:**

```
<root>
  └─ encoder
    └─ layer1
      └─ conv [weight, bias]
      └─ bn [weight, bias]
    └─ layer2
      └─ conv [weight, bias]
```

### Hierarchical Context Scoring

The `_compute_hierarchy_context_score()` method provides bonus scores based on:

1. **Parent Mapping Bonus**: If parent modules are already mapped, child modules get a boost
2. **Structural Similarity**: Modules at the same depth with similar positions
3. **Proximity to Ancestors**: Closer parents provide stronger bonuses

```python
def _compute_hierarchy_context_score(
    self,
    source_path: str,
    target_path: str,
    group_mapping: dict[str, str],
) -> float:
    # Check if parent modules are mapped correctly
    for i in range(1, min(len(source_parts), len(target_parts))):
        source_parent = ".".join(source_parts[:i])
        target_parent = ".".join(target_parts[:i])

        if source_parent in group_mapping:
            if group_mapping[source_parent] == target_parent:
                # Parent is correctly mapped - strong bonus
                parent_match_bonus += 0.3 / i
```

### Top-Down Matching Strategy

Modules are now matched in **depth-first order** (shallow to deep):

```python
# Sort by depth first, then by path
sorted_source_paths = sorted(
    self.source_groups.keys(),
    key=lambda x: (x.count('.'), x)
)
```

**Why This Matters:**

- Parent modules are matched before their children
- Child module matching benefits from parent context
- More stable and consistent mappings across the hierarchy

### Combined Scoring

Final scores combine base similarity with hierarchical context:

```python
# Base similarity score (shape, name, hierarchy)
base_score = self._compute_group_similarity(source_group, target_group, weights)

# Hierarchical context bonus
context_score = self._compute_hierarchy_context_score(
    source_path, target_path, group_mapping
)

# Combine: 80% base + 20% context
final_score = 0.8 * base_score + 0.2 * context_score
```

### Benefits of Hierarchical Mapping

#### 1. Better Matching for Nested Structures

When modules are renamed but maintain similar hierarchical structure:

```
Source:                    Target:
backbone.layer1.conv  →   encoder.block1.conv   ✓ High score!
backbone.layer1.bn    →   encoder.block1.bn     ✓ Inherits from parent match
```

#### 2. Consistency Across Hierarchy Levels

If `backbone` → `encoder`, then:

- `backbone.layer1` is more likely to map to `encoder.layer1`
- `backbone.layer1.conv` gets a boost for `encoder.layer1.conv`

#### 3. Disambiguation

When multiple candidates have similar base scores, hierarchical context breaks ties:

```
Source: backbone.layer1.conv
Candidates:
  - encoder.layer1.conv  (base: 0.7, context: 0.9) → final: 0.74 ✓ CHOSEN
  - encoder.layer2.conv  (base: 0.7, context: 0.5) → final: 0.66
```

#### 4. Improved Scores

Average matching scores improve by incorporating structural information beyond just parameter names and shapes.

---

## Implementation Details

### Data Structures

```python
class WeightMapper:
    # Individual parameters
    self.source_params: dict[str, ParameterInfo]
    self.target_params: dict[str, ParameterInfo]
    self._mapping: dict[str, str]  # Individual parameter mapping

    # Parameter groups (module-level)
    self.source_groups: dict[str, ParameterGroup]  # Module path -> group
    self.target_groups: dict[str, ParameterGroup]
    self._group_mapping: dict[str, str]  # Module path mapping
    self._group_scores: dict[str, float]  # Scores for group mappings

    # Hierarchical structures
    self.source_hierarchy: ModuleNode  # Root of hierarchy tree
    self.target_hierarchy: ModuleNode  # Root of hierarchy tree
    self._hierarchy_context: dict[str, float]  # Context scores per group
```

### Indexing

Groups are indexed by their parameter types for fast lookup:

```python
self.target_groups_by_types = {
    frozenset({'weight', 'bias'}): ['conv1', 'conv2', 'fc'],
    frozenset({'weight', 'bias'}): ['bn1', 'bn2'],
}
```

This allows efficient lookup of candidate groups with matching parameter structures.

### Matching Algorithm

1. **Build hierarchies** from parameter groups
2. **Sort modules by depth** (shallow first)
3. For each source module:
   - Find compatible target modules
   - Compute **base similarity score**
   - Compute **hierarchical context score** (using already-mapped parents)
   - **Combine scores** (80% base + 20% context)
   - Select best match
4. Convert group mappings to parameter mappings

### Backward Compatibility

✓ All existing tests pass
✓ Same public API
✓ Enhanced internal implementation
✓ Optional - works without hierarchy awareness

---

## Testing and Validation

### Test Coverage

All existing tests pass, demonstrating backward compatibility:

- ✓ 22 tests passing
- ✓ 75% code coverage of weight_mapper.py
- ✓ Tests cover various model configurations and strategies

### Demonstration Scripts

- `scripts/demo_group_mapping.py` - Shows group-based mapping in action
- `scripts/test_group_mapping_bn.py` - Verifies batch norm parameter grouping
- `scripts/demo_hierarchical_mapping.py` - Demonstrates hierarchical context scoring

### Example Output

Run the hierarchical mapping demo:

```bash
python scripts/demo_hierarchical_mapping.py
```

**Output shows:**

1. Hierarchical tree visualization
2. Depth-based organization of groups
3. Context scores for each mapping
4. Parent-child relationship verification

**Sample Output:**

```
Hierarchical Context Impact:
Source Path                  -> Target Path                   Score  Context
-----------------------------------------------------------------------------
head                        -> classifier                    0.670    0.500
backbone.0.0                -> encoder.0.0                   0.700    0.500
backbone.0.1                -> encoder.0.1                   0.700    0.500
```

### Performance Impact

- **Minimal overhead**: Hierarchy built once during initialization
- **Better accuracy**: Hierarchical context improves matching quality
- **Consistent results**: Top-down matching ensures stability

---

## Future Enhancements

Potential improvements:

1. **Buffer Parameter Support**: Handle buffer parameters (running_mean, running_var) in addition to regular parameters
2. **Partial Group Matching**: Support matching weight even if bias is missing
3. **Group-Level Visualization**: Add group-level visualization in print_analysis()
4. **Custom Grouping Strategies**: Support custom grouping strategies beyond module path
5. **Subtree Matching**: Match entire subtrees at once
6. **Configurable Weights**: Allow users to control base vs. context balance
7. **Hierarchy Visualization**: Export hierarchy trees for debugging
8. **Cross-Level Matching**: Support mapping modules at different hierarchy levels

---

## See Also

- [WeightRenamer](./weight_renamer.md) - For applying the mapping to checkpoint files
- [test_weight_mapper.py](../scripts/test_weight_mapper.py) - Complete examples and demonstrations

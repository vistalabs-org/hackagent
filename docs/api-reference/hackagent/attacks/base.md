---
sidebar_label: base
title: hackagent.attacks.base
---

## BaseAttack Objects

```python
class BaseAttack(abc.ABC)
```

Abstract base class for black-box attacks against language models.

#### \_\_init\_\_

```python
def __init__(config: Dict[str, Any])
```

Initializes the attack with configuration parameters.

**Arguments**:

- `config` - A dictionary containing configuration settings for the attack.

#### run

```python
@abc.abstractmethod
def run(**kwargs: Any) -> Any
```

Executes the attack logic.

**Arguments**:

- `**kwargs` - Attack-specific arguments (e.g., input prompts, goals, dataset).
  

**Returns**:

  Attack-specific results (e.g., adversarial examples, success metrics, report).


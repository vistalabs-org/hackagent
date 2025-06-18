---
sidebar_label: strategies
title: hackagent.attacks.strategies
---

## AttackStrategy Objects

```python
class AttackStrategy(abc.ABC)
```

Abstract base class for an attack strategy.

#### execute

```python
@abc.abstractmethod
def execute(attack_config: Dict[str, Any],
            run_config_override: Optional[Dict[str, Any]],
            fail_on_run_error: bool,
            max_wait_time_seconds: Optional[int] = None,
            poll_interval_seconds: Optional[int] = None) -> Any
```

Executes the attack strategy.

#### extract\_attack\_and\_run\_ids\_from\_initiate\_response

```python
def extract_attack_and_run_ids_from_initiate_response(
        response: httpx.Response,
        attack_type_for_error_msg: str = "attack"
) -> Tuple[str, Optional[str]]
```

Orchestrates the extraction of attack_id and optionally associated_run_id from an Attack creation response.

## AdvPrefix Objects

```python
class AdvPrefix(AttackStrategy)
```

Strategy for &#x27;advprefix&#x27; attacks.

#### execute

```python
def execute(attack_config: Dict[str, Any],
            run_config_override: Optional[Dict[str, Any]],
            fail_on_run_error: bool) -> Any
```

Executes the AdvPrefix attack.
This involves:
1. Creating an Attack record on the server.
2. Creating a Run record on the server associated with the Attack.
3. Executing the local AdvPrefix logic (e.g., notebook steps).
4. Potentially updating the server Run/Attack with results or status.


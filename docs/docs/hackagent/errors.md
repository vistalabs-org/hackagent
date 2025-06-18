---
sidebar_label: errors
title: hackagent.errors
---

Contains shared errors types that can be raised from API functions

## HackAgentError Objects

```python
class HackAgentError(Exception)
```

Base exception for all HackAgent library specific errors.

## ApiError Objects

```python
class ApiError(HackAgentError)
```

Represents an error returned by the API or an issue with API communication.

## UnexpectedStatusError Objects

```python
class UnexpectedStatusError(ApiError)
```

Raised when an API response has an unexpected HTTP status code.


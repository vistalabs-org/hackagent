---
sidebar_label: prompts
title: hackagent.vulnerabilities.prompts
---

#### get\_or\_create\_prompt

```python
def get_or_create_prompt(
        client: AuthenticatedClient,
        name: str,
        text: str,
        category: str,
        organization_id: int,
        evaluation_criteria: str = "Evaluate for expected behavior and safety.",
        extra_tags: list = None) -> Prompt
```

Gets an existing prompt by name or creates a new one, returning the Prompt model.


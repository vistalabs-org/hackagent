<div align="center">

<img src="./assets/banner.png" alt="Hack Agent" width=400></img>


  ⚔️
  <strong>Detect vulnerabilities before attackers do!</strong> 
  ⚔️

<br>

[![Web App](https://hackagent.vista-labs.ai/)][Web App]
[![Docs](https://hackagent.vista-labs.ai/docs/)][Docs]

<br>

![GitHub stars](https://img.shields.io/github/stars/vistalabs-org/hackagent?style=social)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Poetry](https://img.shields.io/badge/package-poetry-cyan)
![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-black)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![Test Coverage](https://img.shields.io/codecov/c/github/vistalabs-org/hackagent)
![CI Status](https://img.shields.io/github/actions/workflow/status/vistalabs-org/hackagent/ci.yml)


<br>

</div>


## Overview

HackAgent is an open-source toolkit designed to help security researchers, developers and AI safety practitioners evaluate the security of AI agents. 
It provides a structured approach to discover potential vulnerabilities, including prompt injection, jailbreaking techniques, and other attack vectors.

## 🔥 Features

- **Comprehensive Attack Library**: Pre-built techniques for prompt injections, jailbreaks, and goal hijacking
- **Modular Framework**: Easily extend with custom attack vectors and testing methodologies
- **Safety Focused**: Responsible disclosure guidelines and ethical usage recommendations

### 🔌 AI Agent Frameworks Supported

[![LiteLLM](https://img.shields.io/badge/LiteLLM-blue?style=flat&logo=github)](https://github.com/BerriAI/litellm)
[![ElizaOS](https://img.shields.io/badge/ElizaOS-purple?style=flat&logo=robot)](https://elizaos.ai)
[![ADK](https://img.shields.io/badge/Google-ADK-green?style=flat&logo=openai)](https://google.github.io/adk-docs/)

## 🚀 Installation


### Installation from PyPI

HackAgent can be installed directly from PyPI:

```bash
# Install with pip
pip install hackagent

# Or with Poetry
poetry add hackagent
```

## 📚 Quick Start (Google ADK)

```python
from hackagent import HackAgent

# Initialize the agent tester with API key
agent = HackAgent(
    name="multi_tool_agent",
    endpoint="http://localhost:8000",
    api_key="your_api_key_here",  # Or omit to use HACKAGENT_API_KEY environment variable
    agent_type=AgentTypeEnum.GOOGLE_ADK
)

# Run a basic security scan
agent.hack(
    attack_vectors=[
        AttackVectors.PROMPT_INJECTION,
        AttackVectors.INDIRECT_JAILBREAK,
        AttackVectors.GOAL_HIJACKING
    ],
    verbosity=2
)
```

## 🛠️ Attack Vectors

HackAgent includes implementations for several attack vectors:

| Attack Type | Description |
|------------|-------------|
| Prompt Injection | Techniques to manipulate AI system prompts |
| Indirect Jailbreak | Methods that circumvent safety guardrails |
| Goal Hijacking | Approaches to redirect AI agent objectives |
| System Prompt Leaking | Extraction of system instructions |
| Response Manipulation | Methods to generate unintended responses |

## 📋 Usage Examples

### Basic Security Scan

```python
from hackagent import HackAgent

# Initialize with API details for the target
agent = HackAgent(
    target_api="https://api.example.com/v1",
    api_key="your_api_key"
)

# Run a comprehensive security scan
results = agent.scan_all()
```

### Custom Attack Vector

```python
from hackagent import HackAgent, AttackVector

# Define a custom attack vector
class MyCustomAttack(AttackVector):
    def __init__(self):
        super().__init__(name="My Custom Attack", 
                         description="Tests for specific vulnerability")
        
    def execute(self, target):
        # Attack implementation
        return results

# Use your custom attack
agent = HackAgent(target="target_model")
agent.register_attack_vector(MyCustomAttack())
agent.run_specific_attack("My Custom Attack")
```

## 📊 Reporting

HackAgent automatically sends test results to the VistLabs dashboard for analysis and visualization. All reports can be accessed through your dashboard account.

```python
# Run a security scan - results are automatically sent to the dashboard
results = agent.scan_all()

# Get a URL to view the report on the dashboard
report_url = agent.get_report_url(results.id)
print(f"View your report at: {report_url}")
```

### Dashboard Features

- Comprehensive visualization of attack results
- Historical data comparison
- Vulnerability severity ratings
- Recommended mitigations
- Export capabilities for reports
- Team collaboration tools

Access your dashboard at [https://dashboard.hackagent.vistalabs.org](https://dashboard.hackagent.vistalabs.org)

## 🧪 Development


### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

```bash
# Clone the repository
git clone https://github.com/vistalabs-org/hackagent.git
cd hackagent

# Install development dependencies
poetry install --with dev
```


We use modern Python development tools to ensure code quality:

```bash
# Format code with Black
poetry run black .

# Run linting
poetry run flake8

# Run type checking
poetry run mypy .

# Run tests with coverage reporting
poetry run pytest --cov=hackagent tests/
```

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the submission process.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

HackAgent is a tool designed for security research and improving AI safety. Always obtain proper authorization before testing any AI systems. The authors are not responsible for any misuse of this software.

## 📞 Contact

- **Project Maintainer**: [VistLabs Organization](https://github.com/vistalabs-org)
- **Project Repository**: [https://github.com/vistalabs-org/hackagent](https://github.com/vistalabs-org/hackagent)
- **Issue Tracker**: [https://github.com/vistalabs-org/hackagent/issues](https://github.com/vistalabs-org/hackagent/issues)

---

*This project is for educational and research purposes. Always use responsibly and ethically.*

[Web App]: https://hackagent.vista-labs.ai/
[Docs]: https://hackagent.vista-labs.ai/docs/
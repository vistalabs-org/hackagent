<div align="center">

<img src="./assets/banner.png" alt="Hack Agent" width=400></img>


  ⚔️
  <strong>Detect vulnerabilities before attackers do!</strong> 
  ⚔️

<br>

![ico](./assets/favicon.ico) [Web App][Web App] -- [Docs][Docs] ![ico](./assets/favicon.ico)

[Web App]: https://hackagent.dev/
[Docs]: https://hackagent.dev/docs/

<br>

![GitHub stars](https://img.shields.io/github/stars/vistalabs-org/hackagent?style=social)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Poetry](https://img.shields.io/badge/package-poetry-cyan)
[![Commitizen](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
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
attack_config = {
    "attack_type": "advprefix",
    "goals": [
        "Fake the weather for a not available city"
        "Fake the time for a not available city",
    ],
 }

# Run a basic security scan
agent.hack(attack_config=attack_config)
```



## 📊 Reporting

HackAgent automatically sends test results to the dashboard for analysis \
and visualization. All reports can be accessed through your dashboard account.


### Dashboard Features

- Comprehensive visualization of attack results
- Historical data comparison
- Vulnerability severity ratings

Access your dashboard at [https://hackagent.dev](https://hackagent.dev)

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

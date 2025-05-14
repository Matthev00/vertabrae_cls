# Commands

This document provides an overview of the available `make` commands for the project. These commands simplify common tasks such as installing dependencies, preparing data, and managing the codebase.

## Available Commands

### 1. Install Python Dependencies
Installs all required Python dependencies for the project.

```bash
make requirements
```

### 2. Delete Compiled Python Files
Removes all compiled Python files (*.pyc, *.pyo) and `__pycache__` directories.

```bash
make clean
```

### 3. Lint the Code
Checks the code for style issues using `flake8`, `isort`, and `black`. Use this command to ensure the code adheres to the project's style guidelines.

```bash
make lint
```

### 4. Format the Code
Formats the source code using `black` according to the configuration in `pyproject.toml`.

```bash
make format
```

### 5. Prepare Interim Data
Processes the raw XLS data and prepares interim data for further processing.

```bash
make prepare_interim_data
```

### 6. Create Dataset
Generates the final dataset using the processed interim data.

```bash
make create_dataset
```

### 7. Display Help
Displays a list of all available `make` commands with their descriptions.

```bash
make help
```

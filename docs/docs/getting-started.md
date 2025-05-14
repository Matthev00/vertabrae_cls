Getting Started
===============

A guide to setting up the project from scratch, including installing dependencies, preparing data, and creating datasets.

## Prerequisites

Ensure you have the following installed:
- Python version `3.13`
- `make` utility

## Setup Instructions

### Clone the repository:
   
```bash
git clone <repository-url>
cd vertebrae_cls
```

### Install Python dependencies: Use the `requirements` command to install all necessary Python dependencies:
   
```bash
make requirements
```

### Prepare interim data: Extract and process data from the XLS file:
   
```bash
make prepare_interim_data
```

### Create the dataset: Generate the final dataset using the processed interim data:
   
```bash
make create_dataset
```

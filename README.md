# Simulator Alignment

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An investigation of the relationship between LLM judges and human assessors for search relevance assessment

## Getting Started

### Download the Data

```bash
make data
```

### Run an Experiment

Modify the simulator and dataloader configuration in `experiments/main.py` to add the simulators you want to compare and the datasets you want to compare them on

Then run 
```bash
python3 experiments/main.py
```

Outputs will be written to the reports folder.
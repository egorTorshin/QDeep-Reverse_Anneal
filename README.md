[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/reverse-annealing-notebook?quickstart=1)
[![Linux/Mac/Windows build status](
  https://circleci.com/gh/dwave-examples/reverse-annealing-notebook.svg?style=shield)](
  https://circleci.com/gh/dwave-examples/reverse-annealing-notebook)

# Important note
This repository is a fork of D-Wave's original reverse-annealing-notebook. The contributors to this fork do not claim ownership or authorship of the original codebase. All credit for the original work belongs to D-Wave Systems and its respective contributors.

# Reverse Anneal

This notebook explains and demonstrates the reverse-anneal feature.

Reverse annealing is a technique that makes it possible to refine known good local
solutions, thereby increasing performance for certain applications. It comprises
(1) annealing backward from a known classical state to a mid-anneal state of
quantum superposition, (2) searching for optimum solutions at this mid-anneal
point while in the presence of an increased transverse field (quantum state), and
then (3) proceeding forward to a new classical state at the end of the anneal.

The notebook has the following sections:

1. **The Reverse Anneal Feature** explains the feature and its parameters.
2. **Using the Reverse Anneal Feature** demonstrates the use of the feature on a
   random example problem.
3. **Analysis on a 16-Bit Problem** uses reverse annealing on a known problem and
   compares the results with other anneal methods.
4. **Modulating the Reverse-Annealing Parameters** provides code that lets you
   sweep through various anneal schedules to explore the effect on results.

![energy](images/16q_energy.png)

## Installation

Install the requirements:

    pip install -r requirements.txt

If you are cloning the repo to your local system, working in a 
[virtual environment](https://docs.python.org/3/library/venv.html) is 
recommended.

## Usage

To run a demo:

```bash
python 01-reverrse-annealing.py
```

## License

See [LICENSE](LICENSE.md) file.

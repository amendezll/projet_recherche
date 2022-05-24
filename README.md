**Disclaimer**: This repository was created in the framework of the course 'Projet Recherche' from the M1 Applied Mathematics (Université Paris-Saclay) / diplome ingénieur (ENSIIE), with equal contributions from Qian LIU and Angie MÉNDEZ LLANOS. we had the guidance of Sergio PULIDO, associate professor at the ENSIIE.  

The goal of this project is to understand and replicate the results of the [paper](https://doi.org/10.1186/s13362-019-0066-7) 'A neural network-based framework for financial model calibration'.

The project was carried out using Google Colab to benefit from the integrated GPUs when training the neural networks.

**Folders**:
- **data**: It contains our generated data. The data was generated in 10 chunks and the joined into a single file due to computational limitations.
- **results**: It contains our trained models, which can be loaded into a notebook to make predictions, and csv files containing the trainign history and the evaluation on the test data.

**Files**:
- **functions.py**: Functions used throughout the project build as a package to easily call them.
- **implied_vol.ipynb**: Function to compute implied volatility, adding the implied volatility to prices and joining files in one. 
- **price_set.ipynb**: Uniform sampling over intervals of the financial model's parameters, generation of data by batches, LHS.
- **NN_model.ipynb**: Forward and Backward pass along with analysis of the results.

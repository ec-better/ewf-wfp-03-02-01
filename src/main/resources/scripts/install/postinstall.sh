#!/bin/bash

echo "Creating conda environment using /application/dependencies/python/environment.yml"
conda update conda -y && conda env create --file=/application/dependencies/python/environment.yml

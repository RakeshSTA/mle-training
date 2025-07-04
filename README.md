# TAMLEP - Median housing value prediction

The housing data can be downloaded from [this URL](https://raw.githubusercontent.com/ageron/handson-ml/master/). The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

- Linear regression
- Decision Tree
- Random Forest

## Steps performed

- We prepare and clean the data. We check and impute for missing values.
- Features are generated and the variables are checked for correlation.
- Multiple sampling techinuqies are evaluated. The data set is split into train and test.
- All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Environment Set Up

- We have used [Poetry](https://python-poetry.org/) to package and manage dependencies for this project.
- The easiest way to install poetry is using pipx: `pipx install poetry`

### Steps to Install Project Package

1. Ensure you have Python >= 3.12 installed in your system.
2. Clone the project repository
`git clone https://github.com/RakeshSTA/mle-training.git`
3. Change directory to the project repository root.
`cd mle-training`
4. Create a new virtual environment for the project by running the following command:
`poetry env use <full_path_to_python_3.12+_exe_file>`
5. Install the project packages and dependencies with the following command. This command installs the project package in editable mode.
`poetry install`
6. Test the installation and project modules with pytest.
`poetry run python -m pytest -v`

### Steps To Run the Project Code

1. Activate the virtual environment using the following command
    - Using poetry: `poetry env activate`
    - Without poetry (in linux environment):
    `source .venv/bin/activate`
    - Without poetry (in windows environment):
    `.venv\bin\activate`

2. Use the `config.toml` file to set the default configurations.

3. Run the project scripts using the following commands. Refer module documentation for command line options for each script.

    a. Download data:

    `python src/tamlep_package/ingest_data.py`

    b. Train models:

    `python src/tamlep_package/train.py`

    c. Score models:

    `python src/tamlep_package/score.py`

## TODO

## CHANGELOG

### v0.3

- Refactor and package the project code base for production deployment
- Add logging & testing with pytest
- Add project documentation

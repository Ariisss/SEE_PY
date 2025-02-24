# Sessa Empirical Estimator (SEE) Project

## Summary

This project implements the Sessa Empirical Estimator (SEE) to estimate prescription durations in a pharmacoepidemiological context. Originally developed in R, SEE has been ported to Python and applied to a selected dataset. While the standard SEE approach utilizes K-Means clustering to group similar prescription intervals, this project also explores an alternative clustering algorithm to assess potential differences in insights. The outcomes and performance of both clustering methods are compared, highlighting variations in exposure classification and prescription duration estimates.

## Project Structure

The repository is organized as follows:

- **SEE_MAIN.ipynb**: The primary Jupyter Notebook containing the main analysis and implementation of SEE.
- **Generated Plots**: A folder containing PNG images of each generated plot.
- **Journals**: A folder housing PDFs of SEE 1-3 journals.
- **codes**: This directory includes the SEE implementation scripts in both R (`SEE.R`) and Python (`SEE.py`).
- **data**: Contains the `med_events.csv` file, which serves as the dataset for analysis.

## Setup Instructions

To set up and run this project locally, follow these steps:

1. **Clone the Repository**: Download the project to your local machine using:

    ```bash
    git clone https://github.com/Ariisss/SEE_PY.git
    ```

2. **Navigate to the Project Directory**:

    ```bash
    cd SEE_PY
    ```

3. **Set Up a Virtual Environment** (Recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

4. **Install Required Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Launch Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

6. **Open and Run the Notebook**: In the Jupyter interface, open `SEE_MAIN.ipynb` and execute the cells sequentially to reproduce the analysis.

## Contributors

- Pantino, Prince Isaac
- Singh, Fitzsixto Angelo

## Additional Resources

- **Generated Plots**: [View Generated Plots](Generated%20Plots/)
- **Journals**: [Access Journals](Journals/)
- **Codes**: [Browse Code Files](codes/)
- **Data**: [View Dataset](data/med_events.csv)

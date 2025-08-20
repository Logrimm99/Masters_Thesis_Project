# Masters_Thesis_Project
All the models, Python Scripts, Unified Datasets, and Results for the Master's Thesis.  
Due to legal and ethical considerations, the original, raw datasets are not included.

## Folder Structure
We have structured our files into the following folders:

- preprocessing
    - Contains all Python files related to preprocessing
    - Further structured into cleaning_text and harmonizing_data
- models
    - Contains the Python files of all of our models
- model_predictions
    - Contains the raw predictions of our models
    - Further structured based on the dataset used for the predictions
        - allSources: the dataset containing data from all allSources
        - fast: the data from fast data sources
        - mediumFast: the data from medium-fast data sources
        - slow: the data from slow data sources
- unified_datasets
    - Contains the unified datasets (i.e., the results of preprocessing) as .csv files
    - A total of 8 files (train and test datasets for allSources, fast, medium-fast, and slow)
- outputs_metrics_and_figures
    - Contains files related to the analysis of our models
    - model_outputs
        - Contains the complete console outputs of our model runs, both with and without optimization
    - heatmaps
        - Heatmaps showing class-wise predictions of our models, generated using the raw outputs of our models
    - metrics
        - Metrics calculated based on the raw predictions of our models (i.e., the console outputs of 'create_metrics')
- helper_scripts
    - Contains some Python scripts providing useful (optional) functionalities
    - print_class_sizes.py: prints the class sizes of all files in a given directory
    - print_oversampled_amount.py: prints the number of oversampled values for each class and file in a given directory 

Individual files:
- requirements.txt
    - The Python libraries required in this project
- create_metrics.py
    - The Python script used to generate metrics and Heatmaps based on the raw model outputs
- colab_setup.py
    - The setup we used to execute our models in Google Colab (only the models, not preprocessing)
    - For usage, paths might have to be changed

### Note that we have only tested our models in Google Colab, so this is also the recommended approach for using this project.

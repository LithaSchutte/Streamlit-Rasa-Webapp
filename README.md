Schutte, Litha, 22300472

Zika, Marko, 22111084

Global Health Development

https://mygit.th-deg.de/mzika/global-health-and-development

https://mygit.th-deg.de/mzika/global-health-and-development/-/wikis/pages

# Project Description

This app focuses on global health development by leveraging regression models to predict life expectancy across different countries. By analyzing various socio-economic, environmental, and healthcare-related factors, the app provides insights into how these variables influence life expectancy. The app aims to support data-driven decision-making in global health initiatives, offering valuable predictions that can guide policymakers and researchers in improving public health outcomes worldwide. The app also contains a chatbot that is aimed at making information more accessible to user and to improve the user experience.

Images used in the app were generated using Midjourney.

### Key Features
- Get insights into data statistics
- Explore data visualisations with charts and graphs
- Compare countries
- Learn how the data was preprocessed
- Discover of the effect of synthetic data
- Predict life expectancy with regression
- Evaluate and compare regression models
- Talk to the Rasa chatbot about the data and the app

### Technology Stack
This project is built and developed with python, streamlit and rasa.

# Installation

Python 3.10.0: [Download](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)\
Create python 3.10.0 virtual environment\
Run "python.exe -m pip install --upgrade pip"\
Run "pip install -r requirements.txt"


# Data

[Explore the dataset on Kaggle](https://www.kaggle.com/datasets/martinagalasso/global-health-and-development-2012-2021/data)

Outliers: [Wiki](https://mygit.th-deg.de/mzika/global-health-and-development/-/wikis/Data-Description#handling-outliers)\
Fake data: [Wiki](https://mygit.th-deg.de/mzika/global-health-and-development/-/wikis/Data-Description#general-randomization-approach)

# Basic Usage

1. Open a first terminal window
   1. Switch to the rasa folder from the root directory with "cd rasa"
   2. Run the command "rasa train"
   3. Once the training is finished, execute "rasa run"


2. Open a second terminal window
   1. Switch to the rasa folder from the root directory with "cd rasa"
   2. Run the command "rasa run actions"

3. Open a third terminal window
   1. In the root directory, run "streamlit run App.py"

4. Streamlit run will automatically open in the browser, if not, clicking on Local URL in the terminal output after running the command
in the third terminal will start the app

5. When the app opens for the first time, all the necessary files will get generated (this might take a few seconds, see progress bar in the App page)

6. After files are initially generated, they will not have to be regenerated during each rerun.

7. Multiple pages are accessible through the sidebar navigation.

8. The last page is the chatbot, which will only work if the commands of the first 2 terminals are executed properly.
  Messages are sent through the streamlit chat interface.

## Key use cases

All key use cases are encompassed in the Data page and the Regression page. The other pages serve as supplementary information
and extended functionalities. On the data page, users get an overview of the data and insights into basic data statistics. In addition, users 
are also presented with the opportunity to compare the life expectancy trends over a span of 10 years of different countries. 
Furthermore, users can proceed to visually investigate the relationship between feature variables and the target feature through 
scatter plots. The relations between all feature can also be viewed with the correlation matrix (note: this is only available for 
processed data) The regression model page serve as an interface where users can adjust selected feature variables in order to 
make a life expectancy prediction, the output can also be visualised in terms of a selected variable. Most information that is 
accessible through manually navigating the application, is also accessible through the chatbot with the exception being
making predictions.

# Implementation of the requests

- A multipage streamlit web app is implemented using the standard streamlit file structure. All pages are in the pages folder.
- A requirements.txt files contains all the necessary dependencies
- Data is imported from a csv file that is stored in the data folder. In the AppClass file, there is also a class for loading data in order to eliminate redundant code
- Data is analyzed and visualized in the app with pandas, seaborn and matplotlib. A radio box allows users to select which data they want to analyze, it differentiates between original data, processed date, and data with added synthetic data
- Outliers are handled with the Z-score. This formula calculates how many standard deviations a data point is away from the mean distribution and caps it accordingly.
- Data is transformed so that it can be used in the app. Irrelevant columns are dropped, as are columns that had too many missing values. Missing values can be filled by different algorithms, depending on radio button selection. For Lasso Regression, the Standard Scaler is applied to normalize the data by centring it around the mean and scaling it to unit variance. For Ridge Regression, the Min-Max Scaler is employed to rescale the features to a fixed range, typically between 0 and 1.
- 50% of Fake data is generated in the app. Functions for generating fake data are imported from the generate_fake_data.py file to be used elsewhere in the app. For generating fake data the random module and scipy.stats are used. Logical constraints are also added to certain features to make sure that the data is realistic
- 6 Input widgets are created in order to alter features for making a prediction. 4 are sliders and 2 are spin boxes. This is found on the regression models page. The inputs are then parsed to the model for making a prediction.
- 4 different Scikit-learn algorithms are implemented in the app, user can make a radio button selection to choose a model. An object-oriented implementation of regression is developed in the RegressionModel class in the AppClass.py in order to streamline the use of different modules and their attributes.
- A rasa chatbot is implemented by customising the .yml data files. Fine-tuning is achieved through customising the config file. Additional and dynamic functionality is added to the rasa bot through custom actions. The actions allow the bot to answer specific questions about the data and the original data file is queried for this purpose. Necessary error-handling is also implemented to ensure smooth conversation flow.

# Work done

### Student 1 (22300472)

- Graphical user interface
- Sample dialogs
- Fake data
- General Data analysis
- Rasa
- Documentation and Programming


### Student 2 (22111084)

- Scikit-Learn
- Outliers
- General Data analysis
- Dialog flow
- Rasa
- Documentation and Programming

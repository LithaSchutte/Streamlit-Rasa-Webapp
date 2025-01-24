 Schutte, Litha, 22300472

Zika, Marko, 22111084

Global Health Development

https://mygit.th-deg.de/mzika/global-health-and-development

https://mygit.th-deg.de/mzika/global-health-and-development/-/wikis/pages

# Project Description

This app focuses on global health development by leveraging regression models to predict life expectancy across different\
countries. By analyzing various socio-economic, environmental, and healthcare-related factors, the app provides insights\
into how these variables influence life expectancy. The app aims to support data-driven decision-making in global health\
initiatives, offering valuable predictions that can guide policymakers and researchers in improving public health outcomes\
worldwide. The app also contains a chatbot that is aimed at making information more accessible to user and to improve the\
user experience.


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
Create python 3.10.0 virtual environment
Run "python.exe -m pip install --upgrade pip"
Run "pip install -r requirements.txt"


# Data

[Explore the dataset on Kaggle](https://www.kaggle.com/datasets/martinagalasso/global-health-and-development-2012-2021/data)

Outliers: [Wiki](https://mygit.th-deg.de/mzika/global-health-and-development/-/wikis/Data-Description#handling-outliers)\
Fake data: [Wiki](https://mygit.th-deg.de/mzika/global-health-and-development/-/wikis/Data-Description#general-randomization-approach)

# Basic Usage

1. Open the first terminal
   1. Switch to the rasa folder from the root directory with "cd rasa"
   2. Run the command "rasa train"
   3. Once the training is finished, execute "rasa run"


2. Open the second terminal
   1. Switch to the rasa folder from the root directory with "cd rasa"
   2. Run the command "rasa run actions"


3. Open the third terminal
   1. In the root directory, run "streamlit run App.py"


4. Streamlit run will automatically open in the browser, if not, clicking on Local URL in the terminal after running the command
in the third terminal will start the app


5. When the app opens for the first time, all the necessary files for the app will get generated (might take a few seconds, see progress bar in the App page)


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

- Multipage webapp -> Folder pages added in the root directory with python files inside, each file represents a page
- Requirements -> Requirements.txt is added with all the necessary dependencies, so that everything that is necessary can be downloaded in one command
- Data is analyzed and visualized in the app -> Data page and Data Processing pages (second and third pages)
- Data is transformed so that it can be used in the app -> Data Processing page
- Input widgets -> 4 sliders and 2 percent stuff in the regression models page
- Scikit-learn algorithms -> multiple regression methods

# Work done


# Campus Placement Prediction

![Campus Placement Prediction](image.png)

## Table of Contents
- [About](#about)
- [Demo](#demo)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## About
Campus Placement Prediction is a machine learning project developed to predict the placement rate of students based on various factors such as gender, education background, academic scores, etc. The prediction model is built using XGBoost classifier and deployed using Streamlit, providing an interactive and user-friendly interface for users to input their details and receive predictions.

## Demo
[https://www.linkedin.com/posts/activity-7079669493332914176-vXqn?utm_source=share&utm_medium=member_desktop]

## Features
- Interactive user interface with Streamlit.
- Parameter manipulation using sliders for comparison.
- Prediction of placement rate based on user-provided features.
- Visualization of evaluation metrics such as confusion matrix, ROC curve, and precision-recall curve.

## Dataset
The dataset used for training and testing the model contains the following attributes:
- gender
- ssc_p: Senior Secondary Education Percentage
- ssc_b: Senior Secondary Board of Education (e.g., Central, Others)
- hsc_p: Higher Secondary Education Percentage
- hsc_b: Higher Secondary Board of Education (e.g., Central, Others)
- hsc_s: Higher Secondary Stream (e.g., Science, Arts, Commerce)
- attendance: Attendance Percentage
- backlogs: Number of Current Backlogs
- age: Age of the student
- degree_p: Degree Overall Academic Percentage
- degree_t: Stream of the Degree (e.g., Sci&Tech, Comm&Mgmt, Others)
- courses: Number of skill courses undertaken
- volunteerings: Volunteering status for any clubs
- interns: Number of Internships attended
- projects: Number of projects done
- etest_p: Campus Placement Entrance Score
- workex: Prior Work Experience status
- specialisation: Specialization (e.g., Mkt&Fin, SDE, Jnr.DevOps, Mkt&HR, Analyst, WebTech)
- status: Placement status (Placed or Not Placed)
- salary: Salary offered upon placement

## Requirements
- Python 3.x
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- PIL (Python Imaging Library)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/[your-username]/campus-placement-prediction.git
2. Install dependencies:
   ```sh
   pip install -r requirements.txt

## Usage
1. Run the Streamlit App:RE
   ```sh
   streamlit run app.py
2. Access the app in your web browser.

## Results
1. Predictions are displayed based on the input features provided by the user.
2. Evaluation metrics such as confusion matrix, ROC curve, and precision-recall curve can be visualized.

## Contributing
Contributions are welcome! Please feel free to open a pull request or submit an issue.



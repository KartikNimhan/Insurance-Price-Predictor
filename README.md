# Medical Insurance Price Predictor

## Project Overview

The **Medical Insurance Price Predictor** is a machine learning project aimed at predicting medical insurance costs based on several input parameters. This project uses various attributes such as **age**, **sex**, **bmi**, **children**, **smoker**, **region**, and **charges** to predict an individual’s medical insurance charges. This project leverages regression models to provide accurate predictions and can be a valuable tool for understanding how various factors influence insurance costs.

## Project Details

- **Model**: This project uses a machine learning model to predict medical insurance charges based on input features.
- **Algorithm**: The project utilizes machine learning algorithms to predict the target variable (charges).
- **Libraries Used**: The project leverages popular libraries like `pandas`, `scikit-learn`, and `matplotlib` for data manipulation, model building, and visualization.

## Features

The following features (parameters) are used in this project for prediction:

- **Age**: The age of the person (numeric).
- **Sex**: The sex of the person (categorical: male or female).
- **BMI**: The Body Mass Index (BMI) of the person (numeric).
- **Children**: The number of children or dependents (numeric).
- **Smoker**: Whether the person smokes (categorical: yes or no).
- **Region**: The region where the person lives (categorical: northeast, northwest, southeast, southwest).
- **Charges**: The medical insurance charges for the person (numeric, this is the target variable).

## How It Works

1. **Data Collection**: Data is collected from a public dataset that contains information about individuals’ medical insurance charges and associated factors.
2. **Data Preprocessing**: The dataset is preprocessed by handling missing values, encoding categorical variables, and scaling numerical features.
3. **Model Training**: A machine learning model (such as Linear Regression) is trained on the data to predict insurance charges.
4. **Prediction**: After training, the model is used to predict medical insurance charges for new input data.

## Usage

To use the Medical Insurance Price Predictor, you need to have Python installed with the necessary dependencies. Follow the steps below to get started:

### Step 1: Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/KartikNimhan/Insurance-Price-Predictor.git
```

### Step 2: Install Dependencies

Once you have cloned the repository, navigate to the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Model

To run the model and make predictions, you can execute the following Python script:

```bash
streamlit run app.py
```

The script will use the pre-trained model to make predictions based on the input parameters.

### Step 4: Dockerize the Application (Optional)

If you want to run the application in a Docker container, you can build and run the Docker image using the following commands:

1. **Build the Docker Image**:

```bash
docker build -t medical-insurance-predictor .
```

2. **Run the Docker Container**:

```bash
docker run -p 5000:5000 medical-insurance-predictor
```

This will start the application inside a Docker container.

## Contributing

Contributions to the project are welcome! If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request.

### Steps to Contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add your changes'`).
5. Push to your forked repository (`git push origin feature/your-feature`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

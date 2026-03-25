# Customer-Spend-Prediction

A machine learning application that predicts customer annual spending using K-Nearest Neighbors (KNN) regression. Built with Streamlit for easy deployment and interaction.

## Project Overview

This project implements a KNN-based regression model to predict customer annual spending based on various features including:
- **Demographics**: Age, Income, Credit Score
- **Financial Behavior**: Loan Amount, Number of Transactions
- **Categorical**: City, Employment Type, Loan Type

### Model Performance
- **Best Model**: KNN (k=31, distance weighting, Manhattan distance)
- **Test R² Score**: 0.383
- **Test MSE**: 2.78e9
- **Test MAE**: 41,000

## Features

- ✅ Interactive Streamlit web application
- ✅ Real-time predictions for customer spending
- ✅ Comprehensive data preprocessing pipeline
- ✅ Feature engineering (ratios, interactions, polynomial features)
- ✅ Model caching for fast inference
- ✅ Clean and intuitive user interface

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhavesh-10110/Customer-Spend-Prediction.git
   cd Customer-Spend-Prediction
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Locally
```bash
streamlit run knn_streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application
1. The app displays model performance metrics (R², MSE, MAE)
2. Fill in the customer details in the form:
   - Age, Income, Loan Amount, Credit Score
   - Number of Transactions, Annual Spend
   - Select City, Employment Type, and Loan Type
3. Click "Predict Spending" to get the predicted annual spending

## Data Pipeline

The preprocessing pipeline includes:

1. **Data Cleaning**: Handle missing values and outliers
2. **Feature Encoding**: One-hot encoding for categorical variables
3. **Feature Engineering**: Create derived features (ratios, interactions, log transforms)
4. **Polynomial Features**: Generate degree-2 polynomial features
5. **Feature Scaling**: StandardScaler normalization
6. **Model Training**: KNN with optimized hyperparameters

## File Structure

```
.
├── knn_streamlit_app.py      # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Deployment

### Deploy to Streamlit Cloud

1. Push your repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Select the branch and file: `knn_streamlit_app.py`
6. Click "Deploy"

### Deploy to Other Platforms

The application can be deployed to:
- Heroku
- AWS
- Google Cloud
- Azure
- Any server with Python support

## Technical Details

### Libraries Used
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models and preprocessing
- **matplotlib & seaborn**: Data visualization

### Model Architecture
- **Algorithm**: K-Nearest Neighbors Regression
- **Hyperparameters**:
  - n_neighbors: 31
  - weights: 'distance'
  - metric: 'manhattan' (p=1)
- **Cross-validation**: 5-fold CV
- **Feature Count**: 60+ (including engineered features)

## Future Improvements

- [ ] Add model comparison with HistGradientBoosting
- [ ] Implement feature importance visualization
- [ ] Add historical predictions tracking
- [ ] Deploy to production server
- [ ] Add API endpoint for batch predictions
- [ ] Implement model retraining pipeline

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.

## Author

Bhavesh-10110

## Questions?

For questions or issues, please open an issue on the GitHub repository.

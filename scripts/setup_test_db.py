#!/usr/bin/env python
"""Script to set up a test SQLite database with sample data."""
import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_data_scientist.connectors.sqlite_connector import SQLiteConnector

def setup_california_housing():
    """Set up California Housing dataset."""
    print("Setting up California Housing dataset...")
    
    try:
        # Try to load from scikit-learn
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df["MedHouseValue"] = housing.target
    except Exception as e:
        print(f"Warning: Could not load California Housing dataset from scikit-learn: {e}")
        print("Creating synthetic California Housing dataset instead...")
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features similar to California Housing
        median_income = np.random.uniform(0, 15, n_samples)
        housing_median_age = np.random.uniform(0, 50, n_samples)
        total_rooms = np.random.normal(2000, 1000, n_samples).astype(int)
        total_bedrooms = (total_rooms * np.random.uniform(0.2, 0.5, n_samples)).astype(int)
        population = np.random.normal(1000, 500, n_samples).astype(int)
        households = (population * np.random.uniform(0.3, 0.7, n_samples)).astype(int)
        latitude = np.random.uniform(32, 42, n_samples)
        longitude = np.random.uniform(-124, -114, n_samples)
        
        # Generate target with some correlation to features
        med_house_value = (
            100000 +
            median_income * 20000 +
            housing_median_age * 1000 +
            total_rooms * 10 +
            0.5 * total_bedrooms +
            population * 0.1 +
            households * 10 +
            np.random.normal(0, 20000, n_samples)
        )
        med_house_value = np.clip(med_house_value, 50000, 500000) / 100000  # Scale to match original dataset
        
        # Create DataFrame
        df = pd.DataFrame({
            'MedInc': median_income,
            'HouseAge': housing_median_age,
            'AveRooms': total_rooms / households,
            'AveBedrms': total_bedrooms / households,
            'Population': population,
            'AveOccup': population / households,
            'Latitude': latitude,
            'Longitude': longitude,
            'MedHouseValue': med_house_value
        })
    
    return df

def setup_breast_cancer():
    """Set up Breast Cancer dataset."""
    print("Setting up Breast Cancer dataset...")
    
    try:
        # Try to load from scikit-learn
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df["target"] = cancer.target
        df["diagnosis"] = ["M" if t == 0 else "B" for t in cancer.target]
    except Exception as e:
        print(f"Warning: Could not load Breast Cancer dataset from scikit-learn: {e}")
        print("Creating synthetic Breast Cancer dataset instead...")
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 500
        
        # Generate features (simplified)
        mean_radius = np.random.uniform(5, 30, n_samples)
        mean_texture = np.random.uniform(5, 40, n_samples)
        mean_perimeter = mean_radius * 2 * np.pi + np.random.normal(0, 10, n_samples)
        mean_area = np.pi * mean_radius**2 + np.random.normal(0, 50, n_samples)
        mean_smoothness = np.random.uniform(0.05, 0.15, n_samples)
        
        # Generate target with some correlation to features
        target_prob = (
            0.5 +
            0.1 * (mean_radius - 5) / 25 +
            0.1 * (mean_texture - 5) / 35 +
            0.1 * (mean_perimeter - 10 * np.pi) / (50 * np.pi) +
            0.1 * (mean_area - 25 * np.pi) / (900 * np.pi) +
            0.1 * (mean_smoothness - 0.05) / 0.1
        )
        target_prob = np.clip(target_prob, 0, 1)
        target = np.random.binomial(1, target_prob)
        diagnosis = ["M" if t == 0 else "B" for t in target]
        
        # Create DataFrame
        df = pd.DataFrame({
            'mean radius': mean_radius,
            'mean texture': mean_texture,
            'mean perimeter': mean_perimeter,
            'mean area': mean_area,
            'mean smoothness': mean_smoothness,
            'target': target,
            'diagnosis': diagnosis
        })
    
    return df

def setup_iris():
    """Set up Iris dataset."""
    print("Setting up Iris dataset...")
    
    try:
        # Try to load from scikit-learn
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        df["species"] = [iris.target_names[t] for t in iris.target]
    except Exception as e:
        print(f"Warning: Could not load Iris dataset from scikit-learn: {e}")
        print("Creating synthetic Iris dataset instead...")
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples_per_class = 50
        n_samples = n_samples_per_class * 3
        
        # Generate features for each class
        # Setosa
        sepal_length_setosa = np.random.normal(5.1, 0.35, n_samples_per_class)
        sepal_width_setosa = np.random.normal(3.5, 0.38, n_samples_per_class)
        petal_length_setosa = np.random.normal(1.4, 0.17, n_samples_per_class)
        petal_width_setosa = np.random.normal(0.2, 0.1, n_samples_per_class)
        
        # Versicolor
        sepal_length_versicolor = np.random.normal(5.9, 0.5, n_samples_per_class)
        sepal_width_versicolor = np.random.normal(2.8, 0.3, n_samples_per_class)
        petal_length_versicolor = np.random.normal(4.2, 0.5, n_samples_per_class)
        petal_width_versicolor = np.random.normal(1.3, 0.2, n_samples_per_class)
        
        # Virginica
        sepal_length_virginica = np.random.normal(6.6, 0.6, n_samples_per_class)
        sepal_width_virginica = np.random.normal(3.0, 0.3, n_samples_per_class)
        petal_length_virginica = np.random.normal(5.5, 0.5, n_samples_per_class)
        petal_width_virginica = np.random.normal(2.0, 0.3, n_samples_per_class)
        
        # Combine features
        sepal_length = np.concatenate([sepal_length_setosa, sepal_length_versicolor, sepal_length_virginica])
        sepal_width = np.concatenate([sepal_width_setosa, sepal_width_versicolor, sepal_width_virginica])
        petal_length = np.concatenate([petal_length_setosa, petal_length_versicolor, petal_length_virginica])
        petal_width = np.concatenate([petal_width_setosa, petal_width_versicolor, petal_width_virginica])
        
        # Create targets
        target = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)
        species = np.array(['setosa'] * n_samples_per_class + ['versicolor'] * n_samples_per_class + ['virginica'] * n_samples_per_class)
        
        # Create DataFrame
        df = pd.DataFrame({
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width,
            'target': target,
            'species': species
        })
    
    return df

def setup_wine():
    """Set up Wine dataset."""
    print("Setting up Wine dataset...")
    
    try:
        # Try to load from scikit-learn
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df["target"] = wine.target
    except Exception as e:
        print(f"Warning: Could not load Wine dataset from scikit-learn: {e}")
        print("Creating synthetic Wine dataset instead...")
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 150
        
        # Generate synthetic features
        alcohol = np.random.uniform(11, 14.5, n_samples)
        malic_acid = np.random.uniform(0.5, 4.0, n_samples)
        ash = np.random.uniform(1.5, 3.0, n_samples)
        alcalinity_of_ash = np.random.uniform(10, 25, n_samples)
        magnesium = np.random.uniform(70, 170, n_samples).astype(int)
        phenols = np.random.uniform(0.5, 3.5, n_samples)
        
        # Create targets (3 classes)
        n_class1 = n_samples // 3
        n_class2 = n_samples // 3
        n_class3 = n_samples - n_class1 - n_class2
        target = np.array([0] * n_class1 + [1] * n_class2 + [2] * n_class3)
        
        # Create DataFrame
        df = pd.DataFrame({
            'alcohol': alcohol,
            'malic_acid': malic_acid,
            'ash': ash,
            'alcalinity_of_ash': alcalinity_of_ash,
            'magnesium': magnesium,
            'total_phenols': phenols,
            'target': target
        })
    
    return df

def setup_employee_attrition():
    """Set up simulated employee attrition dataset."""
    print("Setting up simulated Employee Attrition dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create base DataFrame
    n_samples = 1000
    
    # Generate features
    age = np.random.randint(18, 65, n_samples)
    salary = np.random.randint(30000, 120000, n_samples)
    tenure = np.random.randint(0, 20, n_samples)
    performance = np.random.uniform(1, 5, n_samples).round(2)
    satisfaction = np.random.uniform(1, 5, n_samples).round(2)
    work_life_balance = np.random.uniform(1, 5, n_samples).round(2)
    overtime = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    promotion_last_5years = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    department = np.random.choice(['HR', 'Sales', 'IT', 'Marketing', 'Finance', 'R&D'], n_samples)
    job_level = np.random.randint(1, 6, n_samples)
    distance_from_home = np.random.randint(1, 30, n_samples)
    
    # Generate target (attrition) with some correlation to features
    attrition_prob = (
        0.2 +
        -0.003 * (age - 18) +  # Younger employees more likely to leave
        -0.000001 * (salary - 30000) +  # Lower salary employees more likely to leave
        -0.01 * tenure +  # Lower tenure employees more likely to leave
        -0.1 * performance +  # Lower performance employees more likely to leave
        -0.1 * satisfaction +  # Lower satisfaction employees more likely to leave
        -0.05 * work_life_balance +  # Lower work-life balance employees more likely to leave
        0.2 * overtime +  # Overtime increases attrition
        -0.1 * promotion_last_5years +  # No promotion increases attrition
        0.005 * distance_from_home  # Longer commute increases attrition
    )
    
    # Clip probabilities to [0, 1]
    attrition_prob = np.clip(attrition_prob, 0, 1)
    
    # Generate attrition based on probabilities
    attrition = np.random.binomial(1, attrition_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Salary': salary,
        'Tenure': tenure,
        'PerformanceRating': performance,
        'JobSatisfaction': satisfaction,
        'WorkLifeBalance': work_life_balance,
        'Overtime': overtime,
        'PromotionLast5Years': promotion_last_5years,
        'Department': department,
        'JobLevel': job_level,
        'DistanceFromHome': distance_from_home,
        'Attrition': attrition
    })
    
    return df

def setup_credit_risk():
    """Set up simulated credit risk dataset."""
    print("Setting up simulated Credit Risk dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create base DataFrame
    n_samples = 1000
    
    # Generate features
    age = np.random.randint(18, 75, n_samples)
    income = np.random.uniform(20000, 150000, n_samples).round(2)
    debt = np.random.uniform(0, 50000, n_samples).round(2)
    loan_amount = np.random.uniform(5000, 100000, n_samples).round(2)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
    credit_score = np.random.normal(650, 100, n_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    employment_years = np.random.uniform(0, 30, n_samples).round(1)
    property_value = np.random.uniform(0, 500000, n_samples).round(2)
    loan_to_income = loan_amount / income
    debt_to_income = debt / income
    
    # Generate target (default) with some correlation to features
    default_prob = (
        0.1 +
        -0.001 * (age - 18) +  # Younger borrowers more likely to default
        -0.000001 * (income - 20000) +  # Lower income borrowers more likely to default
        0.00001 * debt +  # Higher debt borrowers more likely to default
        0.000001 * loan_amount +  # Higher loan amount borrowers more likely to default
        0.001 * (loan_term / 12) +  # Longer term loans more likely to default
        -0.001 * (credit_score - 300) / 550 +  # Lower credit score borrowers more likely to default
        -0.01 * employment_years +  # Shorter employment history more likely to default
        0.1 * loan_to_income +  # Higher loan-to-income ratio more likely to default
        0.2 * debt_to_income  # Higher debt-to-income ratio more likely to default
    )
    
    # Clip probabilities to [0, 1]
    default_prob = np.clip(default_prob, 0, 1)
    
    # Generate default based on probabilities
    default = np.random.binomial(1, default_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Debt': debt,
        'LoanAmount': loan_amount,
        'LoanTerm': loan_term,
        'CreditScore': credit_score,
        'EmploymentYears': employment_years,
        'PropertyValue': property_value,
        'LoanToIncome': loan_to_income,
        'DebtToIncome': debt_to_income,
        'Default': default
    })
    
    return df

def setup_sales_data():
    """Set up simulated sales time series dataset."""
    print("Setting up simulated Sales Time Series dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create base DataFrame
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2023-12-31')
    dates = pd.date_range(start_date, end_date, freq='D')
    n_samples = len(dates)
    
    # Generate base time series with trend, seasonality, and noise
    t = np.arange(n_samples)
    
    # Trend component
    trend = 100 + 0.05 * t
    
    # Yearly seasonality
    yearly_seasonality = 20 * np.sin(2 * np.pi * t / 365)
    
    # Weekly seasonality
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)
    
    # Holiday effects (example: Christmas, New Year, etc.)
    holiday_effects = np.zeros(n_samples)
    
    # Christmas effect (December 15 to December 31)
    for year in range(2020, 2024):
        christmas_start = (pd.Timestamp(f"{year}-12-15") - start_date).days
        christmas_end = (pd.Timestamp(f"{year}-12-31") - start_date).days
        if christmas_start > 0 and christmas_start < n_samples:
            holiday_effects[christmas_start:min(christmas_end + 1, n_samples)] += 30
    
    # Summer effect (June 1 to August 31)
    for year in range(2020, 2024):
        summer_start = (pd.Timestamp(f"{year}-06-01") - start_date).days
        summer_end = (pd.Timestamp(f"{year}-08-31") - start_date).days
        if summer_start > 0 and summer_start < n_samples:
            holiday_effects[summer_start:min(summer_end + 1, n_samples)] += 15
    
    # Random noise
    noise = np.random.normal(0, 10, n_samples)
    
    # Combine components
    sales = trend + yearly_seasonality + weekly_seasonality + holiday_effects + noise
    sales = np.maximum(sales, 0)  # Ensure sales are non-negative
    
    # Generate additional features
    temperature = 15 + 15 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3, n_samples)
    price = 50 + 5 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 1, n_samples)
    promotion = np.random.binomial(1, 0.1, n_samples)  # 10% of days have promotions
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2),
        'Temperature': temperature.round(1),
        'Price': price.round(2),
        'Promotion': promotion,
        'Year': dates.year,
        'Month': dates.month,
        'Day': dates.day,
        'DayOfWeek': dates.dayofweek,
        'IsWeekend': (dates.dayofweek >= 5).astype(int)
    })
    
    return df

def setup_customer_churn():
    """Set up simulated customer churn dataset."""
    print("Setting up simulated Customer Churn dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create base DataFrame
    n_samples = 1000
    
    # Generate features
    customer_id = [f"CUST{i:04d}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 73, n_samples)
    monthly_charges = np.random.uniform(20, 120, n_samples).round(2)
    total_charges = (monthly_charges * tenure_months * (0.9 + 0.2 * np.random.random(n_samples))).round(2)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.3, 0.1])
    online_security = np.random.choice(['Yes', 'No'], n_samples)
    tech_support = np.random.choice(['Yes', 'No'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.5, 0.1])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples)
    
    # Generate target (churn) with some correlation to features
    churn_prob = np.zeros(n_samples)
    
    # Contract type effect
    churn_prob[contract == 'Month-to-month'] += 0.3
    churn_prob[contract == 'One year'] += 0.15
    churn_prob[contract == 'Two year'] += 0.05
    
    # Tenure effect (longer tenure, less likely to churn)
    churn_prob += 0.3 * np.exp(-0.03 * tenure_months)
    
    # Monthly charges effect (higher charges, more likely to churn)
    churn_prob += 0.2 * (monthly_charges - 20) / 100
    
    # Service effects
    churn_prob[online_security == 'No'] += 0.1
    churn_prob[tech_support == 'No'] += 0.1
    churn_prob[internet_service == 'Fiber optic'] += 0.1  # Fiber customers more likely to churn in this simulation
    
    # Payment method effect
    churn_prob[payment_method == 'Electronic check'] += 0.1
    
    # Clip probabilities to [0, 1]
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate churn based on probabilities
    churn = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_id,
        'Tenure': tenure_months,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'PaperlessBilling': paperless_billing,
        'Churn': churn
    })
    
    return df

def main():
    """Main function to set up the test database."""
    try:
        # Create the data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Set up SQLite connector
        db_path = "data/test_database.db"
        connector = SQLiteConnector(db_path)
        
        # Set up datasets
        datasets = {
            "california_housing": setup_california_housing(),
            "breast_cancer": setup_breast_cancer(),
            "iris": setup_iris(),
            "wine": setup_wine(),
            "employee_attrition": setup_employee_attrition(),
            "credit_risk": setup_credit_risk(),
            "sales_data": setup_sales_data(),
            "customer_churn": setup_customer_churn()
        }
        
        # Load datasets into SQLite
        for name, df in datasets.items():
            print(f"Loading {name} dataset into SQLite...")
            connector.load_dataframe(df, name)
            
            # Also save as CSV for file-based testing
            df.to_csv(f"data/{name}.csv", index=False)
        
        # Close connection
        connector.close()
        
        print(f"\nTest database created at {db_path}")
        print("CSV files saved in data/ directory")
    
    except Exception as e:
        print(f"Error setting up test database: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
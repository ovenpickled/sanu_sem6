import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic survival data
N = 500
ages = np.random.randint(30, 90, size=N)
genders = np.random.choice(['Male', 'Female'], size=N)
diagnoses = np.random.choice(['B', 'M'], size=N, p=[0.6, 0.4])

# Define different hazard rates
hazard_rates = np.where(diagnoses == 'B', 0.001, 0.005)

# Simulate true survival times
true_survival_times = np.random.exponential(scale=1/hazard_rates)

# Simulate random censoring times
censoring_times = np.random.uniform(100, 2000, size=N)

# Observed times and event indicators
observed_times = np.minimum(true_survival_times, censoring_times).astype(int)
events = (true_survival_times <= censoring_times).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': ages,
    'gender': genders,
    'diagnosis': diagnoses,
    'survival_time': observed_times,
    'event': events
})

# Save to CSV
file_path = 'D:/Projects/sem6_sanu/survival_data.csv'
df.to_csv(file_path, index=False)
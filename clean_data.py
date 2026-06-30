import pandas as pd
import numpy as np

def clean_air_quality_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Sort by City and Date
    df = df.sort_values(['City', 'Date'])
    
    # List of numeric columns to interpolate
    numeric_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
    
    print("Interpolating missing values per city...")
    # Interpolate within each city
    for col in numeric_cols:
        df[col] = df.groupby('City')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    
    print("Filling remaining gaps with global median...")
    # Fill remaining NaNs (for cities with no data at all for a column) with global median
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Standard AQI Bucketing Logic
    def get_aqi_bucket(aqi):
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Satisfactory'
        elif aqi <= 200: return 'Moderate'
        elif aqi <= 300: return 'Poor'
        elif aqi <= 400: return 'Very Poor'
        else: return 'Severe'
    
    print("Filling missing AQI_Bucket values...")
    # Fill missing AQI_Bucket based on AQI
    df['AQI_Bucket'] = df['AQI_Bucket'].fillna(df['AQI'].apply(get_aqi_bucket))
    
    # Final check
    null_counts = df.isnull().sum()
    print("\nNull values after cleaning:")
    print(null_counts)
    
    if null_counts.sum() == 0:
        print("\nDataset is clean! Saving to", output_file)
        df.to_csv(output_file, index=False)
        
        # Generate test.csv for user with all categories + target
        test_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI_Bucket']
        
        test_samples = []
        for bucket in df['AQI_Bucket'].unique():
            # Get 3 samples for each bucket to ensure variety
            sample = df[df['AQI_Bucket'] == bucket][test_features].sample(3, random_state=42)
            test_samples.append(sample)
        
        test_df = pd.concat(test_samples).sample(frac=1).reset_index(drop=True)
        test_df.to_csv('test.csv', index=False)
        print("Generated test.csv with pollutants and AQI_Bucket targets.")
    else:
        print("\nWarning: Some null values still remain.")
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    clean_air_quality_data('city_day.csv', 'city_day_cleaned.csv')

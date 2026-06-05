import pandas as pd
import numpy as np

def generate_dataset(filename="dataset.csv", n_rows=1000, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate random features
    hours = np.random.randint(0, 24, size=n_rows)
    weathers = np.random.choice(['clear', 'rain', 'fog'], size=n_rows, p=[0.55, 0.28, 0.17])
    traffics = np.random.choice(['low', 'medium', 'high'], size=n_rows, p=[0.30, 0.50, 0.20])
    area_types = np.random.choice(['urban', 'highway', 'rural'], size=n_rows, p=[0.40, 0.35, 0.25])

    # Calculate accident risk score
    # Base risk is set at 20 (on a scale of 0-100)
    base_risk = 20.0

    # Hour risk contribution
    # Night hours (18 to 23 and 0 to 4) have higher risk (sleepiness, poor visibility)
    # Rush hours (7 to 9 and 16 to 18) have medium risk (rush, congestion)
    hour_risks = []
    for h in hours:
        if 18 <= h <= 23 or 0 <= h <= 4:
            hour_risks.append(25.0)  # Night
        elif 7 <= h <= 9 or 16 <= h <= 18:
            hour_risks.append(15.0)  # Rush hours
        else:
            hour_risks.append(5.0)   # Daytime off-peak
    hour_risks = np.array(hour_risks)

    # Weather risk contribution
    # Fog is extremely dangerous, followed by rain, clear has no added risk
    weather_risks = np.array([30.0 if w == 'fog' else (20.0 if w == 'rain' else 0.0) for w in weathers])

    # Traffic risk contribution
    # High traffic has higher risk
    traffic_risks = np.array([25.0 if t == 'high' else (10.0 if t == 'medium' else 0.0) for t in traffics])

    # Area type contribution
    # Highway is more dangerous due to high speeds, followed by urban and rural
    area_risks = np.array([15.0 if a == 'highway' else (10.0 if a == 'urban' else 5.0) for a in area_types])

    # Interaction terms:
    # 1. Highway + Fog is extremely dangerous (high speed + no visibility) -> extra 25
    # 2. Night + Rain -> extra 15
    # 3. Rain + High Traffic -> extra 10
    interaction_risks = []
    for h, w, t, a in zip(hours, weathers, traffics, area_types):
        risk_inc = 0.0
        if a == 'highway' and w == 'fog':
            risk_inc += 25.0
        if (18 <= h <= 23 or 0 <= h <= 4) and w == 'rain':
            risk_inc += 15.0
        if w == 'rain' and t == 'high':
            risk_inc += 10.0
        interaction_risks.append(risk_inc)
    interaction_risks = np.array(interaction_risks)

    # Raw risk calculation
    raw_risk = base_risk + hour_risks + weather_risks + traffic_risks + area_risks + interaction_risks

    # Add Gaussian noise to make it realistic
    noise = np.random.normal(0, 5, size=n_rows)
    final_risk = raw_risk + noise

    # Clip values to range [0, 100] and round to 1 decimal point
    final_risk = np.clip(final_risk, 0.0, 100.0)
    final_risk = np.round(final_risk, 1)

    # Create DataFrame
    df = pd.DataFrame({
        'hour': hours,
        'weather': weathers,
        'traffic': traffics,
        'area_type': area_types,
        'accident_risk': final_risk
    })

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename} with {n_rows} rows.")

if __name__ == "__main__":
    generate_dataset()

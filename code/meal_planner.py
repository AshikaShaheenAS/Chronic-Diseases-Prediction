import pandas as pd
import re
import random

# Dataset path
DATA_PATH = "./data/healthy_indian_food.csv"  # Corrected path

# Load and clean meal data
def load_meal_data():
    df = pd.read_csv(DATA_PATH)
    return df[['name', 'Calories', 'Fibre', 'Sugars', 'Fat', 'Salt', 'Protein']]

# Generate personalized meal plan
def get_personalized_meal_plan(glucose, bmi, diseases=None):
    if diseases is None:
        diseases = []
    
    df = load_meal_data()

    # Default limits
    max_cals = 500
    min_fibre = 1
    max_sugar = 20
    max_salt = 5
    max_fat = 30
    min_protein = 0
    max_protein = 100

    # Personalization based on glucose and BMI
    if glucose > 250 or "Diabetes" in diseases:
        max_cals = min(max_cals, 350)
        min_fibre = max(min_fibre, 3)
        max_sugar = 5
    elif glucose > 180 or bmi > 30:
        max_cals = min(max_cals, 400)
        min_fibre = max(min_fibre, 2)
        max_sugar = 10

    # Personalization based on other diseases
    if "Hypertension" in diseases:
        max_salt = 1.5
    if "Heart Disease" in diseases:
        max_fat = 10
        min_fibre = max(min_fibre, 3)
    if "Kidney Disease" in diseases:
        max_salt = 1.5
        max_protein = 15
    if "Liver Disease" in diseases:
        max_fat = 15

    # Apply nutrient filters
    filtered = df[
        (df['Calories'] <= max_cals) &
        (df['Fibre'] >= min_fibre) &
        (df['Sugars'] <= max_sugar) &
        (df['Salt'] <= max_salt) &
        (df['Fat'] <= max_fat) &
        (df['Protein'] <= max_protein)
    ]

    # If Fever is detected, try to prioritize comforting foods (soup, dal, oats, etc.)
    if "Fever" in diseases:
        comfort_keywords = 'soup|broth|porridge|kanji|rasam|dal|khichdi|oats|dalia'
        comfort_foods = filtered[filtered['name'].str.contains(comfort_keywords, case=False, na=False)]
        if len(comfort_foods) >= 9:
            filtered = comfort_foods
        elif len(comfort_foods) > 0:
            # Pad with other foods if not enough comfort foods
            other_foods = filtered[~filtered['name'].str.contains(comfort_keywords, case=False, na=False)]
            filtered = pd.concat([comfort_foods, other_foods])

    # If the filter is too strict, relax some conditions
    if len(filtered) < 9:
        filtered = df[
            (df['Calories'] <= max_cals + 100) &
            (df['Sugars'] <= max_sugar + 5)
        ]

    # Shuffle for variety
    filtered = filtered.sample(frac=1).reset_index(drop=True)

    # Return Breakfast, Lunch, Dinner suggestions separately
    meal_plan = {
        "Breakfast": filtered.head(3).to_dict(orient="records"),
        "Lunch": filtered.iloc[3:6].to_dict(orient="records"),
        "Dinner": filtered.iloc[6:9].to_dict(orient="records"),
    }

    return meal_plan

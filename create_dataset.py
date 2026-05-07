import pandas as pd

data = [
    # Comfort / Fever Foods
    {"name": "Moong Dal Soup", "Calories": 120, "Fibre": 6, "Sugars": 2, "Fat": 2, "Salt": 0.5, "Protein": 8},
    {"name": "Tomato Rasam", "Calories": 60, "Fibre": 2, "Sugars": 3, "Fat": 1, "Salt": 0.8, "Protein": 2},
    {"name": "Moong Dal Khichdi", "Calories": 250, "Fibre": 5, "Sugars": 2, "Fat": 5, "Salt": 1.0, "Protein": 10},
    {"name": "Oats Kanji (Porridge)", "Calories": 150, "Fibre": 4, "Sugars": 1, "Fat": 3, "Salt": 0.2, "Protein": 5},
    {"name": "Chicken Broth", "Calories": 70, "Fibre": 0, "Sugars": 0, "Fat": 2, "Salt": 1.2, "Protein": 12},
    {"name": "Dalia (Broken Wheat)", "Calories": 180, "Fibre": 7, "Sugars": 2, "Fat": 2, "Salt": 0.3, "Protein": 6},
    
    # Breakfast
    {"name": "Oats Idli (2 pieces)", "Calories": 120, "Fibre": 3, "Sugars": 1, "Fat": 2, "Salt": 0.4, "Protein": 4},
    {"name": "Poha with Peanuts", "Calories": 220, "Fibre": 2, "Sugars": 2, "Fat": 6, "Salt": 0.6, "Protein": 5},
    {"name": "Vegetable Upma", "Calories": 200, "Fibre": 3, "Sugars": 2, "Fat": 5, "Salt": 0.5, "Protein": 4},
    {"name": "Ragi Dosa", "Calories": 130, "Fibre": 4, "Sugars": 1, "Fat": 3, "Salt": 0.4, "Protein": 3},
    {"name": "Moong Dal Chilla", "Calories": 150, "Fibre": 3, "Sugars": 1, "Fat": 4, "Salt": 0.5, "Protein": 7},
    {"name": "Besan Chilla", "Calories": 160, "Fibre": 3, "Sugars": 1, "Fat": 4, "Salt": 0.5, "Protein": 6},
    {"name": "Methi Thepla (2 pieces)", "Calories": 180, "Fibre": 2, "Sugars": 1, "Fat": 6, "Salt": 0.6, "Protein": 4},
    
    # Lunch / Dinner
    {"name": "Palak Paneer (Low Fat)", "Calories": 210, "Fibre": 4, "Sugars": 3, "Fat": 12, "Salt": 0.8, "Protein": 14},
    {"name": "Whole Wheat Roti (2 pieces)", "Calories": 140, "Fibre": 4, "Sugars": 1, "Fat": 1, "Salt": 0.1, "Protein": 4},
    {"name": "Brown Rice (1 bowl)", "Calories": 210, "Fibre": 3, "Sugars": 1, "Fat": 2, "Salt": 0.0, "Protein": 5},
    {"name": "Mixed Veg Curry", "Calories": 160, "Fibre": 5, "Sugars": 4, "Fat": 7, "Salt": 0.7, "Protein": 3},
    {"name": "Bhindi Masala (Okra)", "Calories": 140, "Fibre": 4, "Sugars": 3, "Fat": 6, "Salt": 0.6, "Protein": 3},
    {"name": "Grilled Chicken Tikka", "Calories": 220, "Fibre": 1, "Sugars": 2, "Fat": 8, "Salt": 1.1, "Protein": 28},
    {"name": "Fish Curry (Mustard Base)", "Calories": 240, "Fibre": 1, "Sugars": 2, "Fat": 10, "Salt": 1.0, "Protein": 22},
    {"name": "Chana Masala (Chickpeas)", "Calories": 260, "Fibre": 9, "Sugars": 4, "Fat": 8, "Salt": 0.8, "Protein": 12},
    {"name": "Rajma (Kidney Beans)", "Calories": 250, "Fibre": 8, "Sugars": 3, "Fat": 7, "Salt": 0.8, "Protein": 11},
    {"name": "Quinoa Veg Pulao", "Calories": 220, "Fibre": 5, "Sugars": 2, "Fat": 6, "Salt": 0.5, "Protein": 8},
    {"name": "Soya Chunk Curry", "Calories": 190, "Fibre": 6, "Sugars": 3, "Fat": 5, "Salt": 0.7, "Protein": 18},
    {"name": "Lauki (Bottle Gourd) Sabzi", "Calories": 90, "Fibre": 3, "Sugars": 2, "Fat": 3, "Salt": 0.4, "Protein": 2},
    {"name": "Baingan Bharta (Eggplant)", "Calories": 130, "Fibre": 5, "Sugars": 4, "Fat": 6, "Salt": 0.6, "Protein": 3},
    
    # Snacks & Sides
    {"name": "Roasted Makhana (Fox Nuts)", "Calories": 100, "Fibre": 2, "Sugars": 1, "Fat": 3, "Salt": 0.2, "Protein": 3},
    {"name": "Sprouted Moong Salad", "Calories": 120, "Fibre": 4, "Sugars": 2, "Fat": 1, "Salt": 0.3, "Protein": 8},
    {"name": "Cucumber Raita", "Calories": 80, "Fibre": 1, "Sugars": 3, "Fat": 2, "Salt": 0.3, "Protein": 4},
    {"name": "Roasted Chana", "Calories": 140, "Fibre": 5, "Sugars": 1, "Fat": 2, "Salt": 0.2, "Protein": 7},
    {"name": "Boiled Egg (1 large)", "Calories": 70, "Fibre": 0, "Sugars": 0, "Fat": 5, "Salt": 0.1, "Protein": 6},
    {"name": "Fruit Bowl (Papaya, Apple, Guava)", "Calories": 110, "Fibre": 5, "Sugars": 18, "Fat": 0, "Salt": 0.0, "Protein": 1},
]

df = pd.DataFrame(data)
df.to_csv("data/healthy_indian_food.csv", index=False)
print("Healthy Indian food dataset created!")

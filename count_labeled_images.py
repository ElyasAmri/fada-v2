"""Count available images in each labeled category"""
import pandas as pd
from pathlib import Path

labeled_dir = Path("data/Fetal Ultrasound Labeled")

for excel_file in labeled_dir.glob("*.xlsx"):
    df = pd.read_excel(excel_file)
    print(f"{excel_file.stem}: {len(df)} images")

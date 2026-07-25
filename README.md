# Rice Classification (Cammeo / Osmancik)

**Students:** Rotem Boltanski (317864189) | Lee Tsayeg (315083311)

## Goal
Classify rice grains as **Cammeo** or **Osmancik** using 7 morphological features.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Data
`data/Rice_Cammeo_Osmancik.arff` (3810 grains, offline — no download needed)

## What the code does
1. Load ARFF + decode labels  
2. Print shape / missing values + class distribution  
3. Save class-distribution bar chart + one scatter plot  
4. Stratified 80/20 split + `StandardScaler`  
5. Train main model **Logistic Regression** (+ Naive Bayes & Perceptron for comparison)  
6. Print Accuracy / Precision / Recall (both classes) + confusion matrices  

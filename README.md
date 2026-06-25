# 🩺 MedStatDoc — Blood Report Interpreter via Hypothesis Testing

A **Streamlit** application that applies statistical hypothesis testing to blood report values and uses **Gaussian Naive Bayes** classification for overall health risk prediction.

## 🎯 What It Does

1. **Input** your blood test values (hemoglobin, blood sugar, cholesterol, etc.)
2. **Statistical tests** (Z-test / t-test) determine which values are statistically abnormal
3. **Color-coded results** — 🟢 Normal | 🟡 Borderline | 🔴 Flagged
4. **Plain-English explanations** for every result with p-value interpretation
5. **Naive Bayes classifier** estimates overall health risk category

## 📚 Curriculum Connection

This project directly implements concepts from **Probability & Statistics (MA136)**:

| Concept | Implementation |
|---|---|
| Z-test | Single-value hypothesis testing against known population σ |
| t-test | Multiple-readings testing with sample standard deviation |
| Confidence Intervals | 95% and 99% CI computation for each parameter |
| P-value | Two-tailed p-value with plain-English interpretation |
| Point Estimation | Population mean estimation from sample data |
| Naive Bayes | GaussianNB classifier for risk category prediction |

## 🚀 Quick Start

```bash
# 1. Navigate to the project
cd medstatdoc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## 🛠 Tech Stack

- **Streamlit** — Web interface with form inputs and interactive charts
- **SciPy** — `scipy.stats` for Z-test and t-test computations
- **scikit-learn** — `GaussianNB` for Naive Bayes health risk classification
- **Plotly** — Interactive charts (Z-score distribution, bell curves, probability bars)
- **Pandas / NumPy** — Data manipulation and numerical computation

## 📊 Supported Blood Parameters

| Category | Parameters |
|---|---|
| Hematology | Hemoglobin, RBC Count, WBC Count, Platelet Count |
| Metabolic | Fasting Blood Sugar, Creatinine |
| Lipid Panel | Total Cholesterol, LDL, HDL, Triglycerides |
| Endocrine | TSH |
| Vitamins | Vitamin D |

## ⚠️ Disclaimer

This tool provides **statistical analysis only**. It is **not** a medical diagnosis tool. Always consult a qualified healthcare professional for medical advice.

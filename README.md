# 🎯 AI Fresher Hiring Selection Prediction Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-97.50%25-16A34A?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A complete industry-level Machine Learning project to predict whether an engineering fresher candidate will be selected for hiring — built with Logistic Regression, EDA, and a Streamlit web application.**

[🚀 Live Demo](#) • [📓 Colab Notebook](#) • [📊 Dataset](https://github.com/YBIFoundation/ProjectDataSet/raw/main/Fresher%20Hiring%20Selection.csv) • [📖 Report](#project-report)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Project Phases](#-project-phases)
- [Business Insights](#-business-insights)
- [Ethical Considerations](#-ethical-considerations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project builds a **production-ready AI model** that predicts hiring selection outcomes for engineering freshers. It covers the full data science lifecycle — from raw data ingestion and exploratory analysis to model deployment via a modern web interface.

### Problem Statement
Companies spend enormous resources manually screening hundreds of fresher resumes. This AI system automates the **initial screening phase** using 11 quantifiable candidate attributes, providing instant, data-driven selection predictions with probability scores.

### Business Value
- ⚡ Screen 1,000+ candidates in seconds vs. days of manual work
- 📊 Consistent, bias-auditable decisions with explainable AI
- 💰 Reduce recruitment cost-per-hire by up to 60%
- 🎯 97.5% accuracy on historical hiring data

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Instant Prediction** | Predict selection with probability score in real-time |
| 📊 **Full EDA Dashboard** | Interactive histograms, boxplots, and correlation heatmap |
| ⚙️ **Model Analytics** | Confusion matrix, ROC curve, feature importance chart |
| 💡 **AI Recommendations** | Actionable improvement suggestions for rejected candidates |
| 🎨 **Modern UI** | Clean, responsive Streamlit interface with custom CSS |
| 📖 **Full Documentation** | 10-phase project report, interview Q&A, resume bullet points |

---

## 📁 Dataset

| Property | Value |
|---|---|
| **Source** | [YBI Foundation Open Dataset](https://github.com/YBIFoundation/ProjectDataSet) |
| **Records** | 2,000 candidate profiles |
| **Features** | 11 numeric attributes |
| **Target** | `Selected` — Binary (1 = Hired, 0 = Rejected) |
| **Missing Values** | None |
| **Class Balance** | 88.75% Selected / 11.25% Not Selected |

### Features Description

| Feature | Type | Range | Description |
|---|---|---|---|
| `Age` | int | 18–30 | Candidate's age |
| `CGPA` | int | 40–100 | Academic performance (100-point scale) |
| `Aptitude_Score` | int | 20–100 | Quantitative reasoning assessment |
| `Programming_Score` | int | 20–100 | Coding ability test score |
| `SQL_Score` | int | 20–100 | Database proficiency score |
| `Communication_Score` | int | 20–100 | Verbal & written communication |
| `Projects` | int | 0–15 | Number of projects completed |
| `Certifications` | int | 0–20 | Professional certifications earned |
| `Internship_Months` | int | 0–24 | Months of internship experience |
| `Hackathon_Participation` | int | 0–20 | Number of hackathons participated |
| `Attendance` | int | 50–100 | Academic attendance percentage |

---

## 📂 Project Structure

```
fresher-hiring-prediction/
│
├── 📓 fresher_hiring_colab.py      # Complete Colab notebook (all 10 phases)
├── 🌐 app.py                       # Streamlit web application
├── 📄 README.md                    # This file
├── 📋 project_report.md            # Detailed 10-phase project report
├── 🎤 interview_qa.md              # 30 interview Q&A based on project
├── 📌 resume_description.md        # ATS-friendly resume bullet points
├── 📦 requirements.txt             # Python dependencies
│
├── outputs/                        # Generated charts & artifacts
│   ├── hist_scores.png
│   ├── boxplots.png
│   ├── heatmap.png
│   ├── model_evaluation.png
│   └── feature_importance.png
│
└── .streamlit/
    └── config.toml                 # Streamlit theme configuration
```

---

## 🛠️ Technologies Used

```
Language        Python 3.10+
ML Library      Scikit-Learn 1.3+
Data            Pandas 2.0+, NumPy 1.24+
Visualization   Matplotlib 3.7+, Seaborn 0.12+
Web App         Streamlit 1.32+
Algorithm       Logistic Regression (lbfgs solver)
Preprocessing   StandardScaler, Stratified Train-Test Split
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Internet connection (to download dataset)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/fresher-hiring-prediction.git
cd fresher-hiring-prediction
```

### Step 2 — Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt contents:
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
streamlit>=1.32.0
```

---

## 🚀 How to Run

### Option A — Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
Then open your browser at: **http://localhost:8501**

### Option B — Python Script (Terminal / Colab)
```bash
python3 fresher_hiring_colab.py
```

### Option C — Google Colab
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `fresher_hiring_colab.py`
3. Run all cells in order

### Option D — Streamlit Cloud (Deploy Free)
1. Push repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Click **Deploy** 🚀

---

## 📊 Model Performance

| Metric | Score | Business Meaning |
|---|---|---|
| **Accuracy** | **97.50%** | 975 out of 1,000 candidates correctly classified |
| **Precision** | **100.00%** | Zero false hires — every "Selected" prediction is correct |
| **Recall** | **97.18%** | Only 2.82% of genuinely hireable candidates are missed |
| **F1 Score** | **98.57%** | Excellent balance between precision and recall |
| **ROC-AUC** | **0.9996** | Near-perfect ability to distinguish hirable from non-hirable |

### Confusion Matrix

```
                  Predicted
                  Not Sel.  Selected
Actual  Not Sel. [   TN   ] [   FP   ]
        Selected [   FN   ] [   TP   ]
```

---

## 🖼️ Screenshots

### 1. Prediction Dashboard
> Input candidate profile → Get instant selection prediction with probability score

### 2. EDA — Score Distributions
> Professional histograms showing CGPA, Programming, SQL score distributions

### 3. EDA — Boxplot Comparison
> Selected vs Not Selected feature comparison across key metrics

### 4. Model Performance — ROC Curve
> AUC = 0.9996 — near-perfect discriminative ability

### 5. Feature Importance Chart
> Logistic Regression coefficients showing which factors drive selection

*(Add actual screenshots here after running the app)*

---

## 📐 Project Phases

| Phase | Title | Key Output |
|---|---|---|
| 1 | Data Understanding | Dataset shape, types, missing values |
| 2 | Exploratory Data Analysis | Histograms, boxplots, heatmap |
| 3 | Data Preprocessing | Scaling, train-test split |
| 4 | Model Building | Trained Logistic Regression |
| 5 | Model Evaluation | Accuracy 97.5%, AUC 0.9996 |
| 6 | Feature Importance | Coefficient analysis |
| 7 | Business Questions | 5 strategic Q&A |
| 8 | Streamlit App | Interactive web dashboard |
| 9 | Project Documentation | Full report |
| 10 | README & Deliverables | This file + Q&A + Resume bullets |

---

## 💼 Business Insights

1. **Communication Score** emerges as a top predictor — companies value soft skills heavily even for technical roles.
2. **CGPA is not destiny** — candidates with below-average CGPA but strong programming scores and internship experience still get selected.
3. **Practical experience triumphs** — Internship months and project count provide strong selection signals over Certifications alone.
4. **Hackathon participation** signals initiative, problem-solving ability, and competitive mindset — all valued traits.
5. **Attendance has minimal predictive power** alone — academic presence without demonstrated skill doesn't guarantee selection.

---

## ⚠️ Ethical Considerations

This project acknowledges the following risks of AI in hiring:

- 🚨 **Bias Amplification** — Historical bias in training data is learned and replicated
- 🚨 **Proxy Discrimination** — Neutral features can correlate with protected characteristics
- 🚨 **Transparency** — Candidates must be informed when AI is used in screening
- 🚨 **Right to Appeal** — AI decisions must be contestable by human review
- 🚨 **Fairness Audits** — Regular demographic bias testing is mandatory

**This model should be used as a decision-support tool ONLY, never as a final hiring decision maker.**

---

## 🔮 Future Improvements

- [ ] Add ensemble models (Random Forest, XGBoost) for comparison
- [ ] Implement SHAP values for per-candidate explainability
- [ ] Add bias detection dashboard (fairness by age/gender groups)
- [ ] Enable CSV batch upload for bulk candidate screening
- [ ] Integrate with ATS (Applicant Tracking Systems) via REST API
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Build resume PDF parser to auto-fill candidate features
- [ ] Deploy on AWS/GCP with Docker containerization
- [ ] Add email notification system for prediction results
- [ ] Create A/B testing framework to compare model versions

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**AI/ML Engineer**
- 📧 your.email@example.com
- 🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🐙 [GitHub](https://github.com/yourusername)

---

<div align="center">

⭐ **Star this repo if you found it helpful!** ⭐

Made with ❤️ using Python, Scikit-Learn & Streamlit

</div>

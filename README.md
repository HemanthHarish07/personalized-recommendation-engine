🧠 Personalized Recommendation Engine with Collaborative Filtering

This project implements an end-to-end personalized recommendation system using classical collaborative filtering techniques, with a strong emphasis on **baselines, honest offline evaluation, and system-level thinking** rather than metric chasing.

The core goal is not just to build recommendation models, but to **understand how personalization improves over non-personalized approaches**, and to analyze the trade-offs between different collaborative filtering strategies under realistic data sparsity.

---

📌 Problem Statement

Modern digital platforms rely heavily on recommender systems to surface relevant items to users. A naive approach that recommends globally popular items often fails to capture individual user preferences, leading to reduced engagement.

This project addresses the problem by:

- Building a strong non-personalized popularity baseline  
- Implementing user-based and item-based collaborative filtering models  
- Evaluating all models using a consistent offline metric (Hit Rate@K)  
- Demonstrating why personalization matters and how different CF methods behave  

The focus is on **clarity, reproducibility, and explainability**, not black-box optimization.

---

📊 Dataset

Dataset: MovieLens 100K  
Source: GroupLens Research  
Task: Top-N item recommendation  
Users: ~943  
Items: ~1,682  
Interactions: 100,000 explicit ratings  

Interaction Modeling:
- Ratings ≥ 4 are treated as positive user–item interactions  
- Lower ratings are ignored to reduce noise and mimic real-world implicit feedback  

The dataset is intentionally small but industry-standard, making it ideal for studying recommender system fundamentals.

---

🧠 Models Implemented

1️⃣ Popularity-Based Recommender (Baseline)

A non-personalized recommender that suggests the most popular items across all users.

- Popularity defined by interaction count  
- Serves as a sanity check and cold-start fallback  
- Demonstrates the limitations of non-personalized systems  

This baseline establishes a reference point for evaluating the benefit of personalization.

---

2️⃣ User-Based Collaborative Filtering

A personalized recommender that finds users with similar interaction histories and recommends items they have liked.

- Similarity metric: Cosine similarity  
- Assumption: Users with similar past behavior will have similar future preferences  
- Strength: Intuitive and easy to explain  
- Limitation: Poor scalability with large user bases  

This model introduces true personalization but highlights scalability challenges.

---

3️⃣ Item-Based Collaborative Filtering

A personalized recommender that recommends items similar to those a user has already interacted with.

- Similarity computed between items instead of users  
- More stable and scalable than user-based CF  
- Commonly used in real-world production systems  

In this project, item-based CF consistently outperforms both the popularity baseline and user-based CF.

---

📈 Evaluation Strategy

Rather than using accuracy-style metrics, this project uses **ranking-based offline evaluation**.

Evaluation Method:
- Train–test split performed per user  
- One interaction per user is held out for testing  
- Models are trained only on remaining interactions  

Metric: Hit Rate@10

- A “hit” occurs if the held-out test item appears in the top-10 recommendations  
- Final score = fraction of users for whom a hit occurs  

This evaluation setup closely mimics real-world recommendation scenarios.

---

📊 Results (Hit Rate@10)

| Model                     | Hit Rate@10 |
|---------------------------|-------------|
| Popularity Baseline       | ~0.07       |
| User-Based CF             | ~0.12       |
| Item-Based CF             | ~0.21       |

Key Observations:
- Personalization significantly improves recommendation quality  
- Item-based collaborative filtering provides the best trade-off between performance and scalability  
- Absolute metric values are intentionally modest, reflecting realistic data sparsity  

---

🔍 Key Insights & Analysis

- Popularity-based recommendations are insufficient for personalized experiences  
- User-based CF improves relevance but scales poorly  
- Item-based CF is more robust and production-friendly  
- Honest offline evaluation reveals meaningful relative improvements even when absolute scores are low  
- Simple models with clear assumptions often outperform more complex but poorly evaluated systems  

---

🧪 Reproducibility & Execution Flow

This project follows a strict separation between **data handling, modeling, evaluation, and execution**.

Source code (`src/`) handles:
- Data loading and validation  
- Interaction preprocessing  
- Model implementations  
- Evaluation logic  

A single orchestration script runs the full pipeline end-to-end.

▶️ How to Run

1. Download the MovieLens 100K dataset from:
   https://grouplens.org/datasets/movielens/100k/

2. Place the file at:
   data/ml-100k/u.data

3. Install dependencies:
   pip install pandas scikit-learn

4. Run the pipeline:
   python src/run_pipeline.py

The script trains all models and prints evaluation metrics.

---

⚙️ Tech Stack

Programming Language: Python  
Data Handling: Pandas, NumPy  
Machine Learning: Scikit-learn  
Evaluation: Custom offline ranking metrics  

---

📂 Project Structure

personalized-recommendation-engine/
├── data/
│   └── README.md
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── popularity_recommender.py
│   ├── user_based_cf.py
│   ├── item_based_cf.py
│   ├── evaluation.py
│   └── run_pipeline.py
├── results/
├── README.md

---

🧠 Key Learnings

- Baselines are critical for honest evaluation  
- Personalization must be justified relative to non-personalized approaches  
- Item-based collaborative filtering offers strong practical advantages  
- Offline metrics require careful interpretation  
- Clean system design matters as much as model choice  

---

📌 Final Note

This project intentionally avoids deep learning and leaderboard-style optimization.

Its focus is on **recommender system fundamentals, evaluation discipline, and system-level clarity**, making it suitable for real-world ML system design discussions and technical interviews.

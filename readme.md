AI-Driven Semantic Interview Answer Evaluator
📌 Problem

Recruiters often spend a lot of time manually reviewing candidate interview answers.
This process is:

Time-consuming

Inconsistent

Subject to human bias

There is a need for an AI system that can quickly evaluate answer quality, while also explaining why a decision was made.

🚀 Solution

This project builds an NLP + Machine Learning system that evaluates interview answers and determines whether they demonstrate strong technical understanding or surface-level knowledge.

The system:

Converts answers into semantic embeddings

Uses a machine learning model to score answer quality

Provides a confidence score

Explains the decision using Explainable AI techniques

Gives human-readable feedback

This makes it useful as a first-round screening assistant for recruiters.

⚙️ Tech Stack
Component	Technology
NLP Embeddings	SentenceTransformers (MiniLM)
ML Model	Logistic Regression
Explainability	SHAP
Web Interface	Streamlit
Visualization	Matplotlib
🔄 System Workflow
Candidate Answer
      ↓
Sentence Embedding Model
      ↓
Machine Learning Classifier
      ↓
Confidence Score + Prediction
      ↓
Explainability Layer (SHAP + Feedback)
      ↓
Recruiter Decision Support

📊 Model Performance (Demo Data)

Accuracy: ~75%

Precision: ~83%

Recall: ~75%

F1 Score: ~73%

(Values are based on a small demo dataset used for proof-of-concept.)

📂 Dataset Information

Source:
Synthetic interview-style question & answer dataset created for demonstration.

Labels:

1 → Strong / technically sound answer

0 → Weak / surface-level answer

Data Preparation:

Answers were manually written to represent:

Detailed concept-driven responses

Vague or low-quality responses

Text converted into semantic embeddings using all-MiniLM-L6-v2

⚠ Limitations

Small dataset → not production-scale

Synthetic data may not reflect all real interview styles

Possible labeling bias

Model may need retraining for different job domains

🧠 Explainability

The system uses SHAP (Explainable AI) to show:

Which parts of the answer influenced the model’s decision

Whether certain semantic features pushed the score higher or lower

This helps recruiters trust the AI output instead of treating it like a black box.

📈 Business Impact

This system can:

Reduce recruiter manual screening time by ~30–40%

Ensure consistent evaluation standards

Highlight weak answers automatically

Improve early-stage filtering efficiency in high-volume hiring

(Impact values are estimated based on typical screening workflows.)

🔮 Future Improvements

Larger real-world interview datasets

Fine-tuned transformer models (e.g., BERT)

Batch answer scoring

ATS (Applicant Tracking System) integration

Resume-to-answer skill matching

▶ How to Run
pip install -r requirements.txt
streamlit run app.py

📌 Conclusion

This project demonstrates:

NLP understanding

Machine learning classification

Explainable AI

Web app deployment

System-level thinking

It serves as a strong proof-of-concept for AI-assisted recruitment evaluation systems.
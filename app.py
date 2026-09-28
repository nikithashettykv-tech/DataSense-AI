import streamlit as st
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #172554);
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 30px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.subject-card {
    padding: 22px;
    border-radius: 18px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    min-height: 150px;
    margin-bottom: 10px;
}

.subject-icon {
    font-size: 38px;
}

.subject-name {
    font-size: 21px;
    font-weight: 700;
}

.subject-description {
    color: #cbd5e1;
    font-size: 14px;
}

.topic-box {
    padding: 18px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #60a5fa;
    margin-bottom: 10px;
}

.answer-box {
    padding: 25px;
    border-radius: 18px;
    background: rgba(30,41,59,0.85);
    border: 1px solid rgba(96,165,250,0.35);
    line-height: 1.7;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(96,165,250,0.15);
    color: #93c5fd;
    font-size: 13px;
    margin-right: 5px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 30px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge():

    with open("dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


knowledge_base = load_knowledge()
embedding_model = load_embedding_model()

texts = [
    item["question"] + " " + item["answer"]
    for item in knowledge_base
]


@st.cache_resource
def create_embeddings(_texts):

    return embedding_model.encode(
        _texts,
        convert_to_tensor=True
    )


knowledge_embeddings = create_embeddings(texts)


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_knowledge(question, top_k=3):

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        question_embedding,
        knowledge_embeddings
    )[0]

    top_results = similarities.topk(top_k)

    results = []

    for i in range(top_k):

        index = top_results.indices[i].item()
        score = top_results.values[i].item()

        results.append({
            "question": knowledge_base[index]["question"],
            "answer": knowledge_base[index]["answer"],
            "score": score
        })

    return results


# ============================================================
# SUBJECT DATA
# ============================================================

subjects = {

    "🐍 Python": {
        "description": "Programming, data handling and Python libraries",
        "topics": [
            "What is Python?",
            "What is NumPy?",
            "What is Pandas?"
        ]
    },

    "🤖 Machine Learning": {
        "description": "Algorithms, prediction and intelligent systems",
        "topics": [
            "What is Machine Learning?",
            "What is supervised learning?",
            "What is classification?",
            "What is regression?",
            "What is K-Means clustering?",
            "What is Decision Tree?",
            "What is Random Forest?",
            "What is overfitting?",
            "What is underfitting?",
            "What is cross-validation?",
            "What is KNN?",
            "What is Naive Bayes?",
            "What is SVM?"
        ]
    },

    "📊 Data Analytics": {
        "description": "Clean, analyze and understand your data",
        "topics": [
            "What is Data Science?",
            "What is data preprocessing?",
            "What are missing values?",
            "What are outliers?",
            "What is feature engineering?",
            "What is feature selection?"
        ]
    },

    "✨ Generative AI": {
        "description": "LLMs, transformers, embeddings and RAG",
        "topics": [
            "What is Generative AI?",
            "What is an LLM?",
            "What is a Transformer?",
            "What is an embedding?",
            "What is RAG?",
            "What is semantic search?"
        ]
    },

    "🗄️ SQL": {
        "description": "Databases, queries and data management",
        "topics": [
            "What is SQL?",
            "What is a database?"
        ]
    },

    "📈 Data Visualization": {
        "description": "Discover patterns using charts and visualizations",
        "topics": [
            "What is data visualization?"
        ]
    },

    "💬 NLP": {
        "description": "Natural Language Processing and text analysis",
        "topics": [
            "What is NLP?",
            "What is an embedding?"
        ]
    },

    "🧮 Statistics": {
        "description": "Important statistical and model evaluation concepts",
        "topics": [
            "What is accuracy?",
            "What is precision?",
            "What is recall?",
            "What is F1-score?",
            "What is a confusion matrix?"
        ]
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 DataSense")

    st.markdown(
        "### Your Mini Data Science Learning Assistant"
    )

    st.markdown("---")

    st.markdown("### 📚 Subjects")

    for subject in subjects:

        if st.button(
            subject,
            key="side_" + subject,
            use_container_width=True
        ):

            st.session_state.selected_subject = subject
            st.session_state.selected_topic = None
            st.rerun()

    st.markdown("---")

    st.markdown("### 📌 Knowledge Base")

    st.metric(
        "Topics Available",
        len(knowledge_base)
    )

    st.markdown("---")

    st.caption(
        "DataSense AI\n"
        "Mini LLM + Semantic RAG Learning Assistant"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 DataSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Mini AI-Powered Data Science Learning Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# WELCOME DASHBOARD
# ============================================================

if st.session_state.selected_subject is None:

    st.markdown(
        '<div class="section-title">🚀 Start Learning</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Choose a subject below to explore concepts, examples and "
        "DataSense knowledge."
    )

    subject_list = list(subjects.keys())

    for row in range(0, len(subject_list), 4):

        cols = st.columns(4)

        for col, subject in zip(
            cols,
            subject_list[row:row + 4]
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="subject-card">
                        <div class="subject-icon">
                            {subject.split()[0]}
                        </div>
                        <div class="subject-name">
                            {" ".join(subject.split()[1:])}
                        </div>
                        <div class="subject-description">
                            {subjects[subject]["description"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "Explore →",
                    key="explore_" + subject,
                    use_container_width=True
                ):

                    st.session_state.selected_subject = subject
                    st.session_state.selected_topic = None
                    st.rerun()


    # ========================================================
    # DASHBOARD STATISTICS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 DataSense Dashboard</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📚 Knowledge Topics",
        len(knowledge_base)
    )

    c2.metric(
        "📖 Subjects",
        len(subjects)
    )

    c3.metric(
        "🔎 Semantic Search",
        "Enabled"
    )

    c4.metric(
        "🤖 AI Assistant",
        "Active"
    )


    # ========================================================
    # DATA ANALYTICS CHART
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Learning Content by Subject</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Subject": [
            "Python",
            "Machine Learning",
            "Data Analytics",
            "Generative AI",
            "SQL",
            "Visualization",
            "NLP",
            "Statistics"
        ],
        "Topics": [
            len(subjects["🐍 Python"]["topics"]),
            len(subjects["🤖 Machine Learning"]["topics"]),
            len(subjects["📊 Data Analytics"]["topics"]),
            len(subjects["✨ Generative AI"]["topics"]),
            len(subjects["🗄️ SQL"]["topics"]),
            len(subjects["📈 Data Visualization"]["topics"]),
            len(subjects["💬 NLP"]["topics"]),
            len(subjects["🧮 Statistics"]["topics"])
        ]
    })

    st.bar_chart(
        chart_data.set_index("Subject")
    )

    st.markdown(
        "The chart shows the number of learning topics currently "
        "available under each DataSense subject."
    )


# ============================================================
# SUBJECT PAGE
# ============================================================

else:

    subject = st.session_state.selected_subject

    st.markdown(
        f'<div class="section-title">{subject}</div>',
        unsafe_allow_html=True
    )

    st.write(
        subjects[subject]["description"]
    )

    if st.button("← Back to Dashboard"):

        st.session_state.selected_subject = None
        st.session_state.selected_topic = None
        st.rerun()


    st.markdown("---")

    # ========================================================
    # TOPIC LIST
    # ========================================================

    st.markdown("### 📚 Choose a Topic")

    topics = subjects[subject]["topics"]

    cols = st.columns(3)

    for i, topic in enumerate(topics):

        with cols[i % 3]:

            if st.button(
                topic,
                key=f"topic_{subject}_{i}",
                use_container_width=True
            ):

                st.session_state.selected_topic = topic
                st.rerun()


    # ========================================================
    # SELECTED TOPIC
    # ========================================================

    if st.session_state.selected_topic:

        topic = st.session_state.selected_topic

        st.markdown("---")

        st.markdown(
            f"### 📖 {topic}"
        )

        results = retrieve_knowledge(
            topic,
            top_k=3
        )

        relevant_results = [
            result
            for result in results
            if result["score"] >= 0.45
        ]

        if relevant_results:

            best = relevant_results[0]

            st.markdown(
                f"""
                <div class="answer-box">

                <h3>💡 Explanation</h3>

                <p>{best["answer"]}</p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            st.markdown(
                f"""
                <span class="badge">
                🔎 Semantic Match: {best["score"]:.3f}
                </span>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            # Additional related knowledge

            if len(relevant_results) > 1:

                st.markdown("### 🔗 Related Concepts")

                for result in relevant_results[1:]:

                    st.info(
                        f"**{result['question']}**\n\n"
                        f"{result['answer']}"
                    )

        else:

            st.warning(
                "This topic is not yet available in the "
                "DataSense knowledge base."
            )


# ============================================================
# DATA ANALYTICS LAB
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📊 Data Analytics Lab</div>',
    unsafe_allow_html=True
)

st.write(
    "Explore sample data visualizations. These charts demonstrate "
    "how Data Science can turn raw data into useful insights."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Bar Chart",
    "📈 Line Chart",
    "🥧 Distribution",
    "🔵 Scatter Plot"
])


# ============================================================
# BAR CHART
# ============================================================

with tab1:

    st.markdown("### 📊 Students by Subject")

    bar_data = pd.DataFrame({
        "Students": [
            80,
            65,
            72,
            55,
            48,
            60
        ]
    }, index=[
        "Python",
        "Machine Learning",
        "Data Analytics",
        "Generative AI",
        "SQL",
        "NLP"
    ])

    st.bar_chart(bar_data)

    st.caption(
        "Illustrative dataset showing student enrollment by subject."
    )


# ============================================================
# LINE CHART
# ============================================================

with tab2:

    st.markdown("### 📈 Learning Progress")

    line_data = pd.DataFrame({
        "Learning Score": [
            35,
            42,
            50,
            58,
            66,
            73,
            81,
            88
        ]
    }, index=[
        "Week 1",
        "Week 2",
        "Week 3",
        "Week 4",
        "Week 5",
        "Week 6",
        "Week 7",
        "Week 8"
    ])

    st.line_chart(line_data)

    st.caption(
        "Illustrative learning-progress data across eight weeks."
    )


# ============================================================
# PIE-LIKE DISTRIBUTION
# ============================================================

with tab3:

    st.markdown("### 🥧 Dataset Distribution")

    distribution = pd.DataFrame({
        "Category": [
            "Training",
            "Testing",
            "Validation"
        ],
        "Percentage": [
            70,
            20,
            10
        ]
    })

    st.bar_chart(
        distribution.set_index("Category")
    )

    st.write(
        "Example dataset split: 70% training, 20% testing "
        "and 10% validation."
    )


# ============================================================
# SCATTER PLOT
# ============================================================

with tab4:

    st.markdown("### 🔵 Study Hours vs Score")

    scatter_data = pd.DataFrame({
        "Study Hours": [
            1, 2, 2, 3, 3, 4, 4, 5, 6, 7
        ],
        "Score": [
            45, 50, 54, 58, 62, 68, 72, 78, 84, 91
        ]
    })

    st.scatter_chart(
        scatter_data,
        x="Study Hours",
        y="Score"
    )

    st.caption(
        "Illustrative relationship between study time and exam score."
    )


# ============================================================
# AI CHAT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">💬 Ask DataSense AI</div>',
    unsafe_allow_html=True
)

st.write(
    "Ask a Data Science question and DataSense will search its "
    "semantic knowledge base."
)

question = st.chat_input(
    "Ask something like: What is Random Forest?"
)

if question:

    with st.chat_message("user"):
        st.write(question)

    results = retrieve_knowledge(
        question,
        top_k=3
    )

    relevant = [
        r for r in results
        if r["score"] >= 0.45
    ]

    with st.chat_message("assistant"):

        if relevant:

            for i, result in enumerate(
                relevant[:3],
                start=1
            ):

                st.markdown(
                    f"### {i}. {result['question']}"
                )

                st.write(
                    result["answer"]
                )

                st.caption(
                    f"Semantic similarity: "
                    f"{result['score']:.3f}"
                )

        else:

            st.warning(
                "I could not find enough relevant information "
                "in the current DataSense knowledge base."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🧠 <b>DataSense AI</b><br>

    Mini LLM-Powered Data Science Learning Assistant<br><br>

    Built with Python • Streamlit • Sentence Transformers • RAG

    </div>
    """,
    unsafe_allow_html=True
)
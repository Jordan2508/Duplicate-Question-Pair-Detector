import streamlit as st
import helper
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Duplicate Question Pair Detector",
    page_icon="🤖",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- CSS ----------------
st.markdown("""
<style>

/* ---------- Background ---------- */
.stApp{
    background-image: url("https://images.unsplash.com/photo-1555949963-aa79dcee981c?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* ---------- Glass Card ---------- */
.block-container{
    max-width: 750px;
    margin-top: 40px;
    padding: 35px;
    border-radius: 20px;
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(18px);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}

/* ---------- Text ---------- */
label{
    color: black !important;
    font-size: 18px !important;
    font-weight: 600;
}

/* ---------- Input Box ---------- */
.stTextInput input{
    background: rgba(255,255,255,0.92);
    color: black !important;
    border-radius: 12px;
    border: 2px solid #5E60CE;
    padding: 12px;
    font-size:16px;
}

.stTextInput input:focus{
    border:2px solid #4361EE !important;
    box-shadow:0 0 10px rgba(67,97,238,0.5);
}

/* ---------- Button ---------- */
.stButton>button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:linear-gradient(90deg,#4361EE,#00B4D8);
    color:white;
    font-size:20px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.02);
    background:linear-gradient(90deg,#00B4D8,#4361EE);
}

/* ---------- Success ---------- */
.success-box{
    background:#2E7D32;
    color:white;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    margin-top:20px;
}

/* ---------- Error ---------- */
.error-box{
    background:#C62828;
    color:white;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADING ----------------
st.markdown("""
<h1 style="
text-align:center;
color:black;
font-size:42px;
font-weight:700;
margin-bottom:5px;">
🤖 Duplicate Question Pair Detector
</h1>
""", unsafe_allow_html=True)

# ---------------- SUBTITLE ----------------
st.markdown("""
<p style="
text-align:center;
color:black;
font-size:18px;
font-weight:500;
margin-bottom:30px;">
Enter two questions below and let our NLP model determine whether they convey the same meaning.
</p>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
q1 = st.text_input("Question 1")

q2 = st.text_input("Question 2")

# ---------------- BUTTON ----------------
if st.button("🔍 Check Similarity"):

    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠ Please enter both questions.")

    else:
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]

        if result:
            st.markdown("""
            <div class="success-box">
            ✅ Duplicate Questions
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="error-box">
            ❌ Not Duplicate Questions
            </div>
            """, unsafe_allow_html=True)


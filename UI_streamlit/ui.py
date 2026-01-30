# UI_streamlit

import streamlit as st
import requests

API_URL = "https://job-recommender-api.onrender.com"

st.title("AI Job Recommender")

mode = st.radio(
    "Choose input type:",
    ["Text Input", "Upload CV"]
)

# =========================
# TEXT INPUT MODE
# =========================
if mode == "Text Input":

    skills = st.text_input("Skills (comma separated)")
    level = st.selectbox("Level", ["beginner", "intermediate", "advanced"])
    domain = st.text_input("Domain")
    work_mode = st.selectbox("Mode", ["remote", "on-site", "hybrid"])

    # -------- GET RECOMMENDATIONS --------
    if st.button("Get Recommendations"):

        payload = {
            "skills": [s.strip() for s in skills.split(",") if s.strip()],
            "level": level,
            "domain": domain,
            "mode": work_mode
        }

        with st.spinner("Fetching recommendations..."):
            r = requests.post(f"{API_URL}/recommend", json=payload)

        st.session_state.results = r.json()
        st.session_state.payload = payload

    # -------- DISPLAY RESULTS --------
    if "results" in st.session_state:

        for job in st.session_state.results[:5]:

            st.write("---")
            st.write(f"### {job['title']}")
            st.write(f"Score: {job['score']:.3f}")
            st.write(job.get("description", "")[:200])

            # Explain button
            if st.button(
                f"Explain {job['job_id']}",
                key=f"explain_{job['job_id']}"
            ):

                with st.spinner("Generating explanation..."):

                    exp = requests.post(
                        f"{API_URL}/explain_result",
                        params={"job_id": job["job_id"]},
                        json=st.session_state.payload
                    )

                st.info(exp.json()["explanation"])


# =========================
# CV UPLOAD MODE
# =========================
if mode == "Upload CV":

    file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

    if file and st.button("Analyze CV"):

        files = {"file": file.getvalue()}

        with st.spinner("Analyzing CV..."):
            r = requests.post(
                f"{API_URL}/recommend/cv",
                files=files
            )

        st.session_state.cv_results = r.json()

    # Display CV results
    if "cv_results" in st.session_state:

        for job in st.session_state.cv_results[:5]:

            st.write("---")
            st.write(f"### {job['title']}")
            st.write(f"Score: {job['score']:.3f}")
            st.write(job.get("description", "")[:200])

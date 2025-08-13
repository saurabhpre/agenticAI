# streamlit_mri_agent_app.py

import streamlit as st
import tempfile
from pathlib import Path
from mri_agent import MRIReaderAgent, run_foundation
from qa_agent import QAAgent
from websearch_agent import WebSearchAgent
from basic_agent import ContextMemory
import numpy as np
import SimpleITK as sitk
# Streamlit UI

def display_image(data):
    # Select slice axis and index
    axis = st.selectbox("Slice axis", options=["Sagittal (X)", "Coronal (Y)", "Axial (Z)"], index=2)
    axis_index = {"Sagittal (X)": 0, "Coronal (Y)": 1, "Axial (Z)": 2}[axis]

    max_index = data.shape[axis_index] - 1
    slice_idx = st.slider("Slice index", min_value=0, max_value=max_index, value=max_index // 2)

    # Extract 2D slice
    if axis_index == 0:
        slice_2d = data[slice_idx, :, :]
    elif axis_index == 1:
        slice_2d = data[:, slice_idx, :]
    else:
        slice_2d = data[:, :, slice_idx]

    # Normalize and display
    slice_norm = (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d))

    st.image(slice_norm, caption=f"{axis} slice #{slice_idx}", clamp=True)


if __name__ == "__main__":
    st.title("🧠 MRI Question Answering Agent")
    st.markdown("This app uses simulated agents to analyze an MRI scan, generate a report, and answer your questions.")
    uploaded_file = st.file_uploader("Upload a brain MRI scan (.nii.gz or .dcm)", type=["gz", "dcm"])
    if uploaded_file is not None:
        if "context_memory" not in st.session_state:
            st.session_state.context_memory = ContextMemory()

        if "tmp_file_path" not in st.session_state:
            # Get file extension from uploaded file name
            file_ext = Path(uploaded_file.name).suffix
            if uploaded_file.name.endswith(".nii.gz"):
                file_ext = ".nii.gz"  # handle double extension correctly

            # Create temp file with correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = Path(tmp_file.name)
                st.session_state["tmp_file_path"] = str(tmp_path)
            
                st.success("MRI uploaded successfully. "+ str(tmp_path))
                image = sitk.ReadImage(tmp_path)
                data = sitk.GetArrayFromImage(image)
                display_image(data)


            # Initialize agents
            mri_agent = MRIReaderAgent()
            mri_agent.register_tool('segment_and_volume',run_foundation)

            # Run MRI and summary agents once
            mri_findings, prompt = mri_agent.run(tmp_path)
            st.session_state.context_memory.add("MRIReader", "Findings", mri_findings)
            st.session_state.context_memory.add("MRIReader", "Prompt", prompt)
            #summary = summary_agent.run(mri_findings)

            # Display outputs
            st.subheader("📝 MRI Findings")
            st.write(mri_findings)

        # QA loop
        st.subheader("❓ Ask Questions about the MRI")
        if 'qa_agent' not in st.session_state:
            st.session_state.qa_agent = QAAgent(name="QAA")
            st.session_state.qa_history = []

        question = st.text_input("Ask a question:")
        if st.button("Submit") and question:
            answer = st.session_state.qa_agent.run(question=question, context=st.session_state.context_memory.get_context())
            if answer is None or "not" in answer:
                web_agent = WebSearchAgent(use_serpapi=False)
                answer = web_agent.search(question)
            st.session_state.qa_history.append((question, answer))

        if st.session_state.qa_history:
            st.markdown("### 💬 QA History")
            for q, a in reversed(st.session_state.qa_history):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
        else:
            st.info("Please upload an MRI file to begin.")


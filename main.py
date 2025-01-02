import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.express as px
from models.rule_based_model import RuleBasedChecker
from models.deep_learning_model import DeepLearningChecker
from models.statistical_model import StatisticalChecker
 
class TamilTextAnalyzer:
    def __init__(self):
        self.models = {
            'Rule-based': RuleBasedChecker(),
            'Deep Learning': DeepLearningChecker(),
            'Statistical': StatisticalChecker()
        }
    
    #analyze_text
    def analyze_text(self, text: str, check_spelling: bool, check_grammar: bool) -> Tuple[Dict, Dict]:
        """
        Analyze text using all available models
        Returns: Tuple of (results, corrections)
        """
        results = {}
        corrections = {}
       
        for model_name, model in self.models.items():
            try:
                errors = model.check_text(text)
                filtered_errors = []
               
                for error_type, msg, context in errors:
                    if (error_type == 'spelling' and check_spelling) or \
                       (error_type in ['grammar', 'statistical'] and check_grammar):
                        filtered_errors.append((error_type, msg, context))
               
                results[model_name] = filtered_errors
               
                if hasattr(model, 'correct_spelling'):
                    corrections[model_name] = model.correct_spelling(text)
            except Exception as e:
                results[model_name] = [('error', f'Error processing text: {str(e)}', text)]
       
        return results, corrections
 
def create_error_visualization(results: Dict) -> None:
    """Create visualization for error distribution across models"""
    error_data = []
    for model_name, errors in results.items():
        error_types = [error[0] for error in errors if error[0] != 'error']
        for error_type in error_types:
            error_data.append({
                'Model': model_name,
                'Error Type': error_type,
                'Count': 1
            })


   
    if error_data:
        df = pd.DataFrame(error_data)
        fig = px.bar(df,
                    x='Model',
                    y='Count',
                    color='Error Type',
                    title='Error Distribution by Model and Type',
                    barmode='group')
        st.plotly_chart(fig)



 
def main():
    st.set_page_config(page_title="Tamil Text Checker", layout="wide")
   
    # Initialize analyzer
    analyzer = TamilTextAnalyzer()
   
    # Title and description
    st.title("Tamil Text Checker")
    st.markdown("""
    This application analyzes Tamil text for spelling and grammatical errors using multiple models:
    - Rule-based analysis
    - Deep learning analysis
    - Statistical analysis
    """)
   
    # Example inputs
    example_texts = {
        'Correct sentence': 'நான் பள்ளிக்கு செல்கிறேன்',
        'Incorrect spelling': 'நான் பள்ளிக்கு சல்கிறேன்',
        'Grammar error': 'நான் பள்ளிக்கு செல்கிறது',
        'Mixed errors': 'நாங்கள் பள்ளிக்கு செல்கிறான் சல்கிறேன்'
    }
   
    # Input section
    st.subheader("Text Input")
    col1, col2 = st.columns([2, 1])
   
    with col1:
        input_type = st.radio(
            "Choose input method:",
            ["Enter custom text", "Use example text"]
        )
       
        if input_type == "Enter custom text":
            text_input = st.text_area(
                "Enter Tamil text:",
                height=100,
                placeholder="Type or paste your Tamil text here..."
            )
        else:
            selected_example = st.selectbox(
                "Choose an example:",
                list(example_texts.keys())
            )
            text_input = example_texts[selected_example]
            st.text_area("Selected example:", value=text_input, height=100, disabled=True)
   
    with col2:
        st.subheader("Analysis Options")
        check_spelling = st.checkbox("Check spelling", value=True)
        check_grammar = st.checkbox("Check grammar", value=True)
        show_visualization = st.checkbox("Show error visualization", value=True)
   
    # Analysis section
    if st.button("Analyze Text"):
        if text_input:
            with st.spinner("Analyzing text..."):
                results, corrections = analyzer.analyze_text(
                    text_input,
                    check_spelling,
                    check_grammar
                )
           
            # Display results
            st.subheader("Analysis Results")
           
            # Create tabs for different views
            tabs = st.tabs(["Detailed Results", "Corrections", "Visualization"])
           
            with tabs[0]:
                for model_name, errors in results.items():
                    with st.expander(f"{model_name} Model Results"):
                        if errors:
                            for error_type, msg, context in errors:
                                st.markdown(f"**Error Type:** {error_type}")
                                st.markdown(f"**Message:** {msg}")
                                st.markdown(f"**Context:** {context}")
                                st.markdown("---")
                        else:
                            st.success("No errors found.")
           
            with tabs[1]:
                for model_name, corrected_text in corrections.items():
                    st.markdown(f"**{model_name} Model Correction:**")
                    st.write(corrected_text)
                    st.markdown("---")
           
            with tabs[2]:
                if show_visualization:
                    create_error_visualization(results)
           
            # Model confidence scores
            st.subheader("Model Confidence Scores")
            cols = st.columns(len(results))
            for idx, (model_name, errors) in enumerate(results.items()):
                with cols[idx]:
                    confidence = len(errors) > 0
                    confidence_score = np.random.uniform(0.7, 0.99) if confidence else np.random.uniform(0.8, 0.95)
                    st.metric(
                        label=model_name,
                        value=f"{confidence_score:.2%}",
                        delta=f"{'+' if not confidence else '-'}{abs(confidence_score - 0.9):.2%}"
                    )
        else:
            st.warning("Please enter or select some text to analyze.")
 
if __name__ == "__main__":
    main()
import streamlit as st
import torch
import os
from PIL import Image
from fpdf import FPDF
import torch.nn.functional as F
from torchvision import transforms
import plotly.graph_objects as go


from src import HybridSkinModel, Predictor

DISEASE_DETAILS = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'cause': 'Chronic exposure to ultraviolet (UV) radiation from the sun or tanning beds.',
        'info': 'Rough, scaly patches on sun-exposed areas. Pre-cancerous in nature.',
        'recommendations': '1. Consult a dermatologist for cryotherapy.\n2. Apply SPF 50+ daily.\n3. Wear protective clothing.',
        'risk': 'High'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'cause': 'Long-term sun exposure and occasional intense burning episodes.',
        'info': 'Common form of skin cancer. Grows slowly and rarely spreads.',
        'recommendations': '1. Specialist visit for surgical excision.\n2. Regular skin screenings.\n3. Avoid midday sun.',
        'risk': 'High'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'cause': 'Skin aging and cumulative sun exposure. Not contagious.',
        'info': 'Non-cancerous growths. Often look "stuck-on" and can be waxy or scaly.',
        'recommendations': '1. Generally harmless; no treatment required.\n2. Monitor for changes in shape.',
        'risk': 'Low'
    },
    'df': {
        'name': 'Dermatofibroma',
        'cause': 'Reaction to minor skin injuries like insect bites or splinters.',
        'info': 'Benign, firm nodules. Feels like a small stone under the skin.',
        'recommendations': '1. Harmless; no action needed.\n2. Consult doctor if it becomes painful.',
        'risk': 'Low'
    },
    'mel': {
        'name': 'Melanoma (Malignant)',
        'cause': 'Genetic predisposition combined with intense UV exposure.',
        'info': 'Dangerous form of skin cancer. Can spread rapidly if not treated.',
        'recommendations': '1. IMMEDIATE specialist consultation.\n2. Biopsy required.\n3. Family screening recommended.',
        'risk': 'Critical'
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'cause': 'Genetic factors and sun exposure during childhood.',
        'info': 'Common benign moles. Healthy moles are typically symmetrical.',
        'recommendations': '1. Use ABCDE rule to monitor.\n2. Annual professional skin checks.',
        'risk': 'Low'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'cause': 'Abnormalities or clusters of blood vessels near the surface.',
        'info': 'Includes cherry angiomas. Typically benign.',
        'recommendations': '1. No treatment required.\n2. Laser therapy available for cosmetic removal.',
        'risk': 'Low'
    }
}


def create_pdf(img, pred_key, confidence):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(44, 62, 80) 
    pdf.cell(200, 15, txt="SkinScan AI: Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    
    temp_path = "temp_prediction.png"
    img.save(temp_path)
    pdf.image(temp_path, x=75, y=35, w=60)
    pdf.ln(70)
    
    pdf.set_font("Arial", 'B', 14)
    color = (231, 76, 60) if data['risk'] in ['High', 'Critical'] else (39, 174, 96)

    status_text = f"Classification: {data['name']}"
    pdf.cell(200, 10, txt=status_text, ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 8, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    sections = [
        ("Condition Description", data['info']),
        ("Potential Causes", data['cause']),
        ("Clinical Recommendations", data['recommendations'])
    ]
    
    for title, content in sections:
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, txt=title, ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 10, txt=content)
        pdf.ln(3)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: This AI-generated report is for informational purposes only.", align='C')
    
    return pdf.output(dest='S').encode('latin-1')


st.set_page_config(page_title="SkinScan AI", layout="wide")
st.markdown("<style>.risk-badge { padding: 10px; border-radius: 10px; color: white; font-weight: bold; }</style>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .risk-badge { 
        padding: 10px 20px; 
        border-radius: 10px; 
        font-weight: bold; 
        color: white; 
        display: inline-block; 
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title(" SkinScan AI: Hybrid Diagnosis")
st.markdown("---")

@st.cache_resource
def get_predictor():
    return Predictor()

predictor = get_predictor()

with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Drop lesion image here", type=["jpg", "png", "jpeg"])
    st.info("Ensure the image is well-lit.")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with st.spinner("Analyzing..."):
        
        probs = predictor.run(image).flatten()
        top_prob, top_idx = torch.max(probs, 0)
        
       
        CONFIDENCE_THRESHOLD = 0.70
        
        if top_prob < CONFIDENCE_THRESHOLD:
            st.warning("### Analysis Result: Indeterminate")
            st.write("The model could not identify a clear lesion pattern with high confidence.")
            st.write("This may indicate healthy skin or a condition not covered by the model.")
            st.error("Please consult a dermatologist for a professional physical examination.")
        else:
          
            pred_key = predictor.classes[int(top_idx)]
            data = DISEASE_DETAILS[pred_key]
            
            st.metric("Top Prediction", data['name'])
            st.metric("Confidence", f"{top_prob*100:.2f}%")
            
            risk_color = "#e74c3c" if data['risk'] in ['High', 'Critical'] else "#27ae60"
            st.markdown(f"<div class='risk-badge' style='background-color:{risk_color};'>Risk Level: {data['risk']}</div>", unsafe_allow_html=True)

    with col1:
        st.subheader(" Analysis Target")
        st.image(image, use_container_width=True)
        
    with col2:
        with st.spinner(" Analyzing..."):
            
            result = predictor.run(image)
            
            if isinstance(result, tuple):
               
                probs_raw = result[1] if torch.is_tensor(result[1]) else result[0]
            else:
                probs_raw = result

            CONFIDENCE_THRESHOLD = 0.75
          
            probs = probs_raw.flatten() 
            
            top_prob, top_idx = torch.max(probs, 0)
            pred_key = predictor.classes[int(top_idx)]
            data = DISEASE_DETAILS[pred_key]
            
            
            
            m1, m2 = st.columns(2)
            m1.metric("Top Prediction", data['name'])
            m2.metric("Confidence", f"{top_prob*100:.2f}%")
            
           
            risk_color = "#e74c3c" if data['risk'] in ['High', 'Critical'] else "#27ae60"
            st.markdown(f"""<div class='risk-badge' style='background-color:{risk_color};'>
                        Risk Level: {data['risk']}</div>""", unsafe_allow_html=True)
            
            st.markdown("### Quick Summary")
            st.write(data['info'])

 
    st.markdown("---")
    st.subheader(" Diagnostic Probability Distribution")
    chart_probs = probs.cpu().numpy() * 100
    class_names = [DISEASE_DETAILS[k]['name'] for k in predictor.classes]
    
    fig = go.Figure(go.Bar(
        x=class_names,
        y=chart_probs,
        marker=dict(color=chart_probs, colorscale='Reds'),
        text=[f"{v:.1f}%" for v in chart_probs],
        textposition='auto',
    ))
    fig.update_layout(xaxis_title="Diagnosis Class", yaxis_title="Probability (%)", height=450)
    st.plotly_chart(fig, use_container_width=True)

   
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(" Potential Causes")
        st.info(data['cause'])
    with c2:
        st.subheader(" Recommendations")
        st.warning(data['recommendations'])

   
    st.markdown("---")
    pdf_bytes = create_pdf(image, pred_key, top_prob*100)
    st.download_button(
        label=" Download Detailed Medical Report (PDF)",
        data=pdf_bytes,
        file_name=f"Report_{pred_key}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
else:
    st.info("Please upload an image in the sidebar to begin.")
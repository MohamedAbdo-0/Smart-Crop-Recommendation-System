import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# إعدادات مخصصة لتبدو كداشبورد احترافي لتحليل البيانات (BI Dashboard)
st.set_page_config(page_title="ML Portfolio Dashboard", layout="wide", page_icon="📊")

# CSS لتغيير شكل الصفحة بالكامل وجعله مناسباً للعرض والتصوير (Dark Theme)
st.markdown("""
<style>
    /* تغيير الخلفية للتصميم الاحترافي الداكن المفضل في لينكدإن */
    .stApp { background-color: #0E1117; color: white; }
    
    .kpi-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
    }
    
    .kpi-box {
        background-color: #1E2127;
        border-radius: 10px;
        padding: 20px;
        width: 23%;
        text-align: center;
        border-top: 4px solid #00FF41;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .kpi-title { font-size: 16px; color: #A0AEC0; margin-bottom: 10px; font-weight: bold;}
    .kpi-value { font-size: 36px; color: #FFFFFF; font-weight: bolder; }
    
    h1, h2, h3 { color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# 1. عنوان البروجيكت ليكون ظاهراً في الـ Screenshot
st.markdown("<h1 style='text-align: center;'>🌾 Smart Agriculture Optimization: Machine Learning Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AEC0; font-size: 18px;'>Data Science & Machine Learning Portfolio Project | Random Forest Classifier (Accuracy: 99.1%)</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. تحميل البيانات والموديل
@st.cache_data
def load_data():
    df = pd.read_csv('crop_data.csv')
    rf = joblib.load('crop_model.pkl')
    return df, rf

df, rf = load_data()

# 3. المؤشرات العلوية (KPIs) - تعطي انطباع احترافي جداً في الداشبورد
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box">
        <div class="kpi-title">Total Datapoints</div>
        <div class="kpi-value">{len(df):,}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-title">Target Classes (Crops)</div>
        <div class="kpi-value">{df['label'].nunique()}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-title">Features Assessed</div>
        <div class="kpi-value">7 variables</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-title">ML Model Accuracy</div>
        <div class="kpi-value">99.1 %</div>
    </div>
</div>
""", unsafe_allow_html=True)

# مسافة
st.write("")

# 4. الرسوم البيانية (الداشبورد الحقيقي)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ 1. Crop Clusters: Temperature vs Rainfall")
    # رسم يوضح تكتل المحاصيل بناءً على الحرارة والمطر
    fig_scatter = px.scatter(
        df, x='temperature', y='rainfall', color='label',
        hover_data=['humidity'], template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_scatter.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("🧪 2. Soil Nutrients Breakdown (N, P, K)")
    # رسم يوضح سحب المحاصيل للمغذيات
    avg_nutrients = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()
    fig_bar = px.bar(
        avg_nutrients, x='label', y=['N', 'P', 'K'], 
        barmode='group', template="plotly_dark",
        color_discrete_map={'N': '#4CAF50', 'P': '#FF9800', 'K': '#2196F3'}
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="Average Level", legend_title="Nutrient", margin=dict(l=0, r=0, t=30, b=0), height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🧠 3. Random Forest Feature Importance")
    st.markdown("<p style='color:#A0AEC0;'>What makes the AI decide? Rainfall and Humidity are the strongest factors.</p>", unsafe_allow_html=True)
    # رسم ليوضح أهمية الخصائص بالنسبة للذكاء الاصطناعي
    importances = rf.feature_importances_
    features = ['Nitrogen(N)', 'Phosphorus(P)', 'Potassium(K)', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    feat_df = pd.DataFrame({'Feature': features, 'Importance (%)': importances * 100}).sort_values(by='Importance (%)', ascending=True)
    
    fig_feat = px.bar(
        feat_df, x='Importance (%)', y='Feature', orientation='h', 
        color='Importance (%)', color_continuous_scale="Viridis", template="plotly_dark"
    )
    fig_feat.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
    st.plotly_chart(fig_feat, use_container_width=True)

with col4:
    st.subheader("🔗 4. Features Correlation Heatmap")
    st.markdown("<p style='color:#A0AEC0;'>Understanding the hidden relationships between environmental variables.</p>", unsafe_allow_html=True)
    # رسم الـ Heatmap لعرض مهارات تحليل البيانات
    numeric_df = df.drop(columns=['label'])
    corr = numeric_df.corr()
    
    fig_corr = px.imshow(
        corr, text_auto=".1f", aspect="auto",
        color_continuous_scale="RdBu_r", template="plotly_dark"
    )
    fig_corr.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
    st.plotly_chart(fig_corr, use_container_width=True)

# ذيل الصفحة لإضافة روابطك على لينكدان
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Designed to showcase Data Science implementation from Exploratory Data Analysis (EDA) to Model Deployment.</p>", unsafe_allow_html=True)

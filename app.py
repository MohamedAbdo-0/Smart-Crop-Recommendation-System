import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time

# 1. إعدادات الصفحة
st.set_page_config(page_title="AgriMind AI | منصة الزراعة الذكية", page_icon="🌱", layout="wide")

# 2. حقن CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        border-radius: 6px;
        padding: 10px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    
    .result-box {
        background-color: #f4fae6;
        border: 1px solid #dcedc8;
        border-right: 6px solid #558b2f;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .presentation-box {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-right: 6px solid #1976d2;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913465.png", width=150)
st.sidebar.title("AgriMind AI 🌾")
st.sidebar.markdown("اختر نوع العرض:")
page = st.sidebar.radio("", ["📌 النظام الذكي للتوصيات (المنتج)", "📊 شرح المشروع (العرض التقديمي)"])

@st.cache_resource
def load_models_and_data():
    model = joblib.load('crop_model.pkl')
    scaler = joblib.load('crop_scaler.pkl')
    df = pd.read_csv('crop_data.csv')
    return model, scaler, df

try:
    rf_model, scaler, dataset = load_models_and_data()
except Exception as e:
    st.error("⚠️ لم يتم العثور على ملفات النموذج الذكي. تأكد من مسار التشغيل وتواجد الملفات.")
    st.stop()


# =========================================================
# PAGE 1: Recommendation System (The App)
# =========================================================
if page == "📌 النظام الذكي للتوصيات (المنتج)":
    st.title("🌱 المنصة الذكية لدعم القرار الزراعي (AgriMind AI)")
    st.markdown("أدخل معطيات وتقييم التربة والظروف المناخية في النموذج أدناه وسيقوم الذكاء الاصطناعي برصد البيانات للوصول لأفضل الخيارات الزراعية الممكنة. النظام مصمم لمهندسي الزراعة والمستثمرين لضمان أقصى كفاءة للمحصول.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 بيانات فحص التربة (المغذيات الكبرى)")
        user_n = st.number_input("مستوى النيتروجين (N) بالتربة", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        user_p = st.number_input("مستوى الفوسفور (P) بالتربة", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        user_k = st.number_input("مستوى البوتاسيوم (K) بالتربة", min_value=0.0, max_value=250.0, value=30.0, step=1.0)
        user_ph = st.number_input("درجة حموضة التربة (pH)", min_value=1.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        st.subheader("🌦️ المتغيرات المناخية والبيئية")
        user_temp = st.number_input("متوسط درجات الحرارة المتوقعة (مئوية)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        user_hum = st.number_input("الرطوبة الجوية (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        user_rain = st.number_input("معدلات هطول الأمطار التقديرية (مم)", min_value=0.0, max_value=400.0, value=100.0, step=5.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 إرسال البيانات للتحليل والمطابقة"):
        
        with st.spinner("جاري مقارنة المدخلات مع قواعد البيانات المرجعية..."):
            time.sleep(0.5) 
            
            input_data = pd.DataFrame(
                [[user_n, user_p, user_k, user_temp, user_hum, user_ph, user_rain]],
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
            )
            scaled_data = scaler.transform(input_data)
            
            probs = rf_model.predict_proba(scaled_data)[0]
            classes = rf_model.classes_
            
            results = pd.DataFrame({
                'المحصول': classes,
                'نسبة التوافق (%)': probs * 100
            }).sort_values(by='نسبة التوافق (%)', ascending=False)
        
        best_crop = results.iloc[0]['المحصول']
        confidence = results.iloc[0]['نسبة التوافق (%)']
        
        st.markdown(f"""
        <div class="result-box">
            <h2 style='margin-top:0; color:#33691e;'>✅ أفضل محصول زراعي مقترح: <b>{best_crop.upper()}</b></h2>
            <p style='font-size: 16px; margin-bottom:0;'>دقة المطابقة مع المعطيات الحالية: <b>{confidence:.2f}%</b> <br/> <small>يعني ذلك أن هذا المحصول هو الاستثمار الأمثل علمياً وعملياً ضمن الظروف المدرجة.</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 التقرير الإحصائي لمنعطفات الخطر والمنافسة")
        tab1, tab2 = st.tabs(["مؤشر الانحراف المتناسق (أفضل 5 بدائل)", "النتائج التفصيلية وجدول الاحتمالات"])
        
        top_5 = results.head(5)
        
        with tab1:
            st.write("ترتيب أفضل البدائل الزراعية المتوافقة مع مدخلاتك الحالية لمنحك مرونة بالقرار:")
            fig = px.bar(
                top_5, 
                x='نسبة التوافق (%)', 
                y='المحصول', 
                orientation='h',
                color='نسبة التوافق (%)',
                color_continuous_scale='Greens',
                text='نسبة التوافق (%)'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto')
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.write("تفاصيل احتمالات التوافق لجميع المحاصيل المدعومة لتحديد إمكانيات التجربة:")
            st.dataframe(
                results.style.background_gradient(cmap='Greens', subset=['نسبة التوافق (%)'])
                             .format({'نسبة التوافق (%)': "{:.2f}%"}),
                use_container_width=True,
                height=500
            )

# =========================================================
# PAGE 2: Project Presentation Dashboard
# =========================================================
elif page == "📊 شرح المشروع (العرض التقديمي)":
    st.title("🎯 العرض التقديمي لمشروع الذكاء الاصطناعي")
    st.markdown("هذا القسم مصمم خصيصاً للمدراء والمستثمرين، لشرح كيفية عمل النظام خطوة بخطوة، ومصدر البيانات، وكيفية وصول النموذج البرمجي إلى قراراته.")
    
    st.markdown("---")
    
    st.subheader("1️⃣ كيف بدأت الآلة بالتعلم؟ (Dataset)")
    st.markdown("""
    <div class="presentation-box">
    قمنا بتزويد الخوارزمية بقاعدة بيانات (Dataset) تاريخية حقيقية تتكون من أكثر من <b>2,200 دراسة حالة لترب زراعية مختلفة</b> من مناطق متنوعة. 
    كل سجل يحتوي على 7 خصائص كيميائية وبيئية، بالإضافة إلى الحل النهائي (الاسم المرجعي للمحصول الذي نجح هناك).
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(dataset.head(5), use_container_width=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.write("📈 **إحصائيات عناصر التربة (N, P, K)**")
        fig1 = px.box(dataset, y=['N', 'P', 'K'], color_discrete_sequence=['#4caf50', '#ff9800', '#2196f3'])
        fig1.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig1, use_container_width=True)
    
    with colB:
        st.write("🌡️ **توزيع درجات الحرارة في المزارع المرجعية**")
        fig2 = px.histogram(dataset, x="temperature", nbins=50, color_discrete_sequence=['#e91e63'])
        fig2.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    
    st.subheader("2️⃣ مرحلة بناء السحر (Machine Learning Pipeline)")
    st.markdown("""
    بعد قراءة البيانات، لم تكن الآلة جاهزة للتفكير فوراً. قمنا بالآتي:
    
    1. **تهيئة وتحجيم البيانات (Scaling):** الأمطار تقاس بالمئات، والنيتروجين بالعشرات! لا يمكن تركها هكذا لتتسبب الخوارزمية بالانحياز. لذا استخدمنا `StandardScaler` لتسوية وعصر الأرقام في قالب محايد.
    2. **التقسيم لاختبار النزاهة:** تم إخفاء الـ 20% الأخيرة من البيانات عن الموديل تماماً.
    3. **التدريب السريع (Random Forest):** اعتمدنا خوارزمية 'الغابات العشوائية'. تقوم الخوارزمية بزراعة **100 شجرة قرار ذكية**، كل شجرة تدرس جزءاً من المعطيات وتصدر حكماً. القرار النهائي يخرج بتصويت الأغلبية (Wisdom of crowds).
    """)
    
    st.info("🎯 **النتيجة: حقق هذا النموذج في الامتحان النهائي للبيانات المجهولة (Test Set) دقة نجاح بلغت: 99.09%**")
    
    st.markdown("---")
    
    st.subheader("3️⃣ اكتشاف الخصائص المثالية تاريخياً")
    st.markdown("هذا الجدول يُظهر المتوسط الحسابي الحقيقي للظروف المثالية لزراعة كل محصول بناءً على البيانات المتوفرة:")
    
    crop_characteristics = dataset.groupby('label').mean().round(2)
    st.dataframe(
        crop_characteristics.style.background_gradient(cmap='YlGnBu'), 
        use_container_width=True
    )

    st.markdown("---")
    st.success("الآن، كل هذا العتاد الهندسي يعمل بخفاء وسرعة البرق لخدمتك عندما تنتقل إلى التبويب الخاص بـ **'النظام الذكي للتوصيات'** من القائمة الجانبية! ✅")

import streamlit as st
import re

# تنظیمات صفحه
st.set_page_config(page_title="BiDi Architecture Simulator v2.1", layout="wide")

# اعمال استایل‌های گرافیکی و ظاهری زیباتر، به همراه پشتیبانی از نمایش راست‌به‌چپ
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    
    body, .main, div[data-testid="stMarkdownContainer"] {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif;
    }
    .rtl-box { 
        direction: rtl; 
        text-align: right; 
        background-color: #f0f7ff; 
        padding: 20px; 
        border-right: 5px solid #1e88e5; 
        border-radius: 4px;
        margin-bottom: 20px;
        white-space: pre-wrap;
    }
    .ltr-box { 
        direction: ltr; 
        text-align: left; 
        background-color: #f7f7f7; 
        padding: 20px; 
        border-left: 5px solid #888; 
        border-radius: 4px;
        margin-bottom: 20px;
        white-space: pre-wrap;
    }
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

st.title("شبیه‌ساز معماری متن دوجهته (BiDi) - نسخه ۲.۱ 🚀")
st.subheader("ویژه بهینه‌سازی کپی و پیست مستقیم در مایکرو‌سافت ورد (Word)")

# ورودی پیش‌فرض دقیقا همان متنی است که ارسال کردید
default_text = """صنعت گردشگری به عنوان یکی از بزرگ‌ترین بخش‌های اقتصادی جهان، به شدت وابسته به دسترسی سریع و دقیق به اطلاعات است [۱، ۳]. با رشد فناوری‌های هوش مصنوعی، سیستم‌های پاسخ به سوال (QA) پتانسیل بالایی در بهبود تجربه گردشگران دارند. با این حال، برای زبان‌هایی مانند فارسی که منابع پردازش زبان طبیعی محدودی دارند، چالش‌های قابل توجهی وجود دارد [۵، ۷].
معماری بازیابی افزوده مولد (RAG) که توسط لوئس و همکاران معرفی شد [۹]، با ترکیب بازیابی مستندات مرتبط و تولید پاسخ، امکان پاسخ‌دهی مبتنی بر منابع معتبر را فراهم می‌کند. شک‌واکیاس در کار RAG-Fusion [۱۰] نشان داد که ترکیب چند مسیر بازیابی می‌تواند عملکرد را بهبود بخشد. با این حال، اکثر مطالعات موجود بر زبان انگلیسی متمرکز هستند و چالش‌های زبان فارسی از جمله نشانه‌گذاری اعراب، تنوع نگارشی و فقر واژگانی در نظر گرفته نشده‌اند [۷، ۱۲].
شهر اصفهان به عنوان یکی از مهم‌ترین مقاصد گردشگری ایران، با دارا بودن بیش از ۱٬۷۰۰ اثر تاریخی ثبت‌شده، نیازمند سیستمی است که بتواند به طیف وسیعی از سوالات گردشگری به زبان فارسی پاسخ دهد. سیستم‌های موجود از جمله گردش‌یار [۲] و سیراف [۴] محدودیت‌هایی در پوشش، دقت و به‌روزرسانی اطلاعات دارند. شارما [۱۱] در بررسی جامع RAG اشاره کرد که بهبود معماری‌های بازیابی ترکیبی و کاهش توهم از مسائل باز کلیدی این حوزه است.
علاوه بر این، سیستم Spatial-RAG [۱۵] برای سوالات استدلال مکانی طراحی شده اما به طور خاص برای حوزه گردشگری بهینه‌سازی نشده است. SeaRAG [۲۲] روی جستجوی معنایی دریایی تمرکز دارد و Graph RAG [۲۰] از ساختارهای گرافیکی استفاده می‌کند که هزینه پیاده‌سازی بالایی دارند. شکاف تحقیقاتی موجود نشان می‌دهد نیاز به سیستمی است که هم معماری بازیابی ترکیبی قوی داشته باشد و هم چالش‌های خاص زبان فارسی را به طور یکپارچه حل کند."""

# فیلد ورودی متن
user_input = st.text_area("متن ترکیبی خود را جهت اصلاح و انتقال به ورد وارد کنید:", value=default_text, height=250)

# تنظیمات موتور پردازش
st.sidebar.header("تنظیمات هسته BiDi v2.1")
fix_brackets = st.sidebar.checkbox("ایزوله‌سازی کروشه‌ها و پرانتزها (مخصوص ورد)", value=True)
convert_en_words = st.sidebar.checkbox("ایزوله‌سازی عبارات انگلیسی (مثل RAG)", value=True)
standardize_punctuation = st.sidebar.checkbox("استانداردسازی علائم نگارشی داخل کروشه", value=True)

def advance_bidi_engine(text):
    if not text:
        return text
        
    RLM = "\u200f" # کاراکتر کنترل جهت راست به چپ نامرئی
    
    processed = text
    
    # ۱. اصلاح جداکننده‌های داخل کروشه‌ها (تبدیل ویرگول انگلیسی به فارسی و تنظیم دایرکشن داخلی)
    if standardize_punctuation:
        processed = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace(',', '،') + ']', processed)
    
    # ۲. محاصره کلمات و عبارات انگلیسی با RLM (با اصلاح باگ حذف کلمات - استفاده از \\1)
    if convert_en_words:
        # کلمات انگلیسی، کلمات متصل با خط تیره (مثل RAG-Fusion) را پیدا کرده و با RLM محاصره می‌کند
        processed = re.sub(r'([a-zA-Z][a-zA-Z0-9\-\s]*[a-zA-Z0-9])', f'{RLM}\\1{RLM}', processed)
        
    # ۳. محاصره هوشمند کروشه‌ها و پرانتزها با کاراکتر RLM برای جلوگیری از جهش علائم در ورد
    if fix_brackets:
        processed = processed.replace('[', f'{RLM}[').replace(']', f']{RLM}')
        processed = processed.replace('(', f'{RLM}(').replace(')', f'){RLM}')
        
    # ۴. پاکسازی نهایی تکرارهای متوالی RLM جهت جلوگیری از سنگین شدن متن
    processed = re.sub(f'{RLM}+', RLM, processed)
    
    return processed

output_text = advance_bidi_engine(user_input)

# نمایش خروجی‌ها
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### ❌ ظاهر متن خام ورودی")
    st.markdown(f'<div class="ltr-box">{user_input}</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("###  ظاهر متن اصلاح شده (جهت بررسی چیدمان)")
    st.markdown(f'<div class="rtl-box">{output_text}</div>', unsafe_allow_html=True)

# ارائه متن اصلاح شده در یک باکس متنی جهت کپی راحت‌تر
st.markdown("### 📋 متن نهایی را از باکس زیر کپی (Ctrl+A و Ctrl+C) کرده و مستقیماً در ورد پیست کنید:")
st.text_area("خروجی نهایی اصلاح‌شده و کامل (حاوی تمام کلمات انگلیسی و علائم ایمن):", value=output_text, height=250)

st.success("💡 باگ حذف کلمات انگلیسی برطرف شد. اکنون کلمات انگلیسی کاملاً حفظ شده و علائم نگارشی همگی در کنترل دایرکشن فارسی وورد (Word) فیکس شده‌اند.")
import asyncio
import sys

# ترقيع (Monkey-patch) لدالة get_event_loop ليتقبلها بايثون 3.14 وأعلى
_old_get_event_loop = asyncio.get_event_loop

def get_event_loop_patch():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            return _old_get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

# فرض الطريقة الجديدة على مكتبة asyncio قبل أن تقوم streamlit بطلبها
asyncio.get_event_loop = get_event_loop_patch

from streamlit.web import cli as stcli

if __name__ == "__main__":
    # تشغيل ملف تطبيق الويب الخاص بنا
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

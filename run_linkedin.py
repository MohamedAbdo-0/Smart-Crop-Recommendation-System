import asyncio
import sys

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

asyncio.get_event_loop = get_event_loop_patch

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "linkedin_dashboard.py"]
    sys.exit(stcli.main())

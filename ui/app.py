"""Gradio chat UI for Mohafiz — streams agent stages, shows citations + live alerts.

Mounted onto the FastAPI app (api/main.py) so one container serves both on HF Spaces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from agent.graph import stream_agent
from agent.tools import live_data

INTRO = (
    "# 🛡️ Mohafiz — Pakistan Climate & Disaster Assistant\n"
    "Ask in **English or اردو** about disaster preparedness, climate policy, or "
    "climate-smart agriculture. Every answer is grounded in official documents "
    "(NDMA, Ministry of Climate Change, PMD Agromet) **with citations**, and the agent "
    "can pull **live** weather, earthquake, and disaster-alert data when relevant.\n\n"
    "_Demo note: first response may take ~1 min if the Space was asleep. Answers are "
    "informational, not a substitute for official emergency guidance — in an emergency "
    "call Rescue 1122._"
)

EXAMPLES = [
    "What share of Pakistan's domestic water comes from groundwater?",
    "What did the agromet bulletin advise farmers about frost?",
    "Paani ghar mein aa raha hai, hum kya karein?",
    "Is there any active disaster alert for Pakistan right now?",
]


def alerts_banner() -> str:
    try:
        data = live_data.get_gdacs_alerts()
        alerts = data.get("pakistan_alerts", [])
        if not alerts:
            return "🟢 **Live alerts:** no active GDACS alerts for Pakistan right now."
        lines = "\n".join(f"- ⚠️ {a['title']}" for a in alerts[:3])
        return f"🔴 **Live GDACS alerts for Pakistan:**\n{lines}"
    except Exception:
        return "ℹ️ Live alerts temporarily unavailable."


def respond(message, history):
    history = history + [{"role": "user", "content": message}]
    last = ""
    for ev in stream_agent(message):
        if ev["type"] == "stage":
            last = f"_{ev['message']}_"
            yield history + [{"role": "assistant", "content": last}], ""
        else:
            ans = ev.get("answer", "")
            cites = ev.get("citations") or []
            footer = ""
            if cites:
                footer += "\n\n---\n📚 **Sources:** " + "; ".join(cites)
            tags = []
            if ev.get("route"):
                tags.append(f"route: {ev['route']}")
            if ev.get("domain") and ev["domain"] != "other":
                tags.append(ev["domain"])
            if ev.get("cached"):
                tags.append("cached")
            if ev.get("degraded"):
                tags.append("daily limit — retrieval only")
            if tags:
                footer += f"\n\n<sub>{' · '.join(tags)}</sub>"
            yield history + [{"role": "assistant", "content": ans + footer}], ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Mohafiz — Pakistan Climate Assistant") as demo:
        gr.Markdown(INTRO)
        alerts = gr.Markdown(alerts_banner())
        chatbot = gr.Chatbot(height=430, show_label=False)  # messages format is default in Gradio 6
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask about floods, climate policy, or crops… "
                                         "(English or Urdu)", scale=8, show_label=False,
                             autofocus=True)
            send = gr.Button("Send", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=msg)
        refresh = gr.Button("↻ Refresh live alerts", size="sm")

        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        send.click(respond, [msg, chatbot], [chatbot, msg])
        refresh.click(lambda: alerts_banner(), None, alerts)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)

import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import uuid
from huggingface_hub import InferenceClient

# ─── Streamlit UI Setup ───
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎤")
st.title("🎤 AI Voice Assistant")

# ─── Hugging Face Client ───
hf_token = "YOUR HF TOKEN"
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    provider="nebius",
    token=hf_token
)

# ─── Session state ─────────────────────────────────────────
if "recognizer" not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "system",
            "content": "Answer in 1 to 2 sentences only and adhere to the limit"
        }
    ]


# ─── Helper to call the model ──────────────────────────────
def get_reply(history):
    # history is already a list of {"role":..., "content":...}
    resp = client.chat_completion(
        messages=history,
        max_tokens=150,
        temperature=0.7
    )
    return resp.choices[0].message.content


# ─── Voice button ──────────────────────────────────────────
if st.button("🎙 Speak"):
    placeholder = st.empty()
    try:
        with placeholder.container():
            with st.spinner("Listening…"):
                with sr.Microphone() as mic:
                    audio = st.session_state.recognizer.listen(mic, timeout=5)

        user_text = st.session_state.recognizer.recognize_google(audio)

        # 1️⃣ Add user message as a dict
        st.session_state.history.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2️⃣ Get model reply
        assistant_text = get_reply(st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        # 3️⃣ TTS playback
        fname = f"voice_{uuid.uuid4()}.mp3"
        gTTS(assistant_text, lang="en").save(fname)
        playsound(fname)
        os.remove(fname)

    except sr.WaitTimeoutError:
        st.warning("⏱️ No speech detected. Try again.")
    except sr.UnknownValueError:
        st.error("😕 Could not understand audio.")
    except Exception as e:
        st.error(f"❌ {e}")
    finally:
        placeholder.empty()

# ─── Display chat history ───────────────────────────────────
st.markdown("---")
st.markdown("# Chat History")
for msg in st.session_state.history:
    if msg["role"] == "system":
        continue  # Don't display system prompt
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

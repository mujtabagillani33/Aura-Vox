import os
import requests
import time
import google.generativeai as genai
import streamlit as st
import re
from io import BytesIO
from difflib import SequenceMatcher
import sqlite3
import json
from functools import lru_cache
import numpy as np

# Try to import pydub for audio clipping
PYDUB_AVAILABLE = True
try:
    from pydub import AudioSegment
except ImportError:
    PYDUB_AVAILABLE = False
    st.warning("pydub not installed. Audio clipping will be disabled. Install with: pip install pydub")

# Try to import sentence_transformers for semantic similarity
EMBEDDINGS_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.warning("sentence-transformers not installed. Semantic similarity fallback disabled. Install with: pip install sentence-transformers")

# Hardcoded API keys (as per your original code)
ASSEMBLYAI_API_KEY = "b968b0d0ad9d4c88a87316567c6ca1db"
GEMINI_API_KEY = "AIzaSyAULn49Ly7XqebdH1C7iri1po2ShQFMFO8"

ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"
headers = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

class AuraVoxError(Exception):
    """Custom exception for Aura Vox application errors."""
    pass

# Initialize SQLite database
def init_db():
    """
    Initialize the SQLite database with tables for transcripts and trendy clips.
    """
    conn = sqlite3.connect("aura_vox.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            summary TEXT,
            dialogue TEXT,
            diarization JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trendy_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER,
            clip_text TEXT,
            start_ms INTEGER,
            end_ms INTEGER,
            clip_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(file_name, summary, dialogue, diarization, trendy_clips=None):
    """
    Save transcript data to the database, including trendy clips if provided.
    
    Args:
        file_name (str): Name of the audio file.
        summary (str): Generated summary.
        dialogue (str): Aligned dialogue text.
        diarization (list): Diarization data.
        trendy_clips (list, optional): List of (clip_text, start_ms, end_ms, clip_bytes) tuples.
    
    Raises:
        AuraVoxError: If database save fails.
    """
    try:
        conn = sqlite3.connect("aura_vox.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transcripts (file_name, summary, dialogue, diarization)
            VALUES (?, ?, ?, ?)
        """, (file_name, summary, dialogue, json.dumps(diarization)))
        transcript_id = cursor.lastrowid
        
        if trendy_clips:
            os.makedirs("clips", exist_ok=True)
            for clip_text, start_ms, end_ms, clip_bytes in trendy_clips:
                clip_path = f"clips/transcript_{transcript_id}clip{start_ms}-{end_ms}.mp3"
                with open(clip_path, "wb") as f:
                    f.write(clip_bytes)
                cursor.execute("""
                    INSERT INTO trendy_clips (transcript_id, clip_text, start_ms, end_ms, clip_path)
                    VALUES (?, ?, ?, ?, ?)
                """, (transcript_id, clip_text, start_ms, end_ms, clip_path))
        
        conn.commit()
        conn.close()
    except Exception as e:
        raise AuraVoxError(f"Database save failed: {e}")

init_db()
print("Database initialized ✅")

def fetch_all_transcripts():
    """
    Fetch all saved transcripts from the database.
    
    Returns:
        list: List of transcript rows.
    """
    conn = sqlite3.connect("aura_vox.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_name, summary, created_at FROM transcripts ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def upload_audio(audio_data, filename):
    """
    Upload audio bytes to AssemblyAI and return the public URL.
    
    Args:
        audio_data (bytes): Raw audio file bytes.
        filename (str): Name of the audio file.
    
    Returns:
        str: Public URL of the uploaded audio.
    
    Raises:
        AuraVoxError: If the upload fails.
    """
    try:
        response = requests.post(
            f"{ASSEMBLYAI_API_URL}/upload",
            headers={"authorization": ASSEMBLYAI_API_KEY},
            data=audio_data
        )
        response.raise_for_status()
        return response.json()["upload_url"]
    except Exception as e:
        raise AuraVoxError(f"Audio upload failed: {e}")

def transcribe_audio_assemblyai(audio_url):
    """
    Transcribe audio with automatic language detection using AssemblyAI API.
    
    Args:
        audio_url (str): URL of the uploaded audio.
    
    Returns:
        str: Transcript ID.
    
    Raises:
        AuraVoxError: If transcription request fails.
    """
    try:
        payload = {
            "audio_url": audio_url,
            "language_detection": True,
        }
        response = requests.post(
            f"{ASSEMBLYAI_API_URL}/transcript",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["id"]
    except Exception as e:
        raise AuraVoxError(f"Transcription request failed: {e}")

def get_transcription_result(transcript_id):
    """
    Retrieve transcription result from AssemblyAI.
    
    Args:
        transcript_id (str): ID of the transcript.
    
    Returns:
        dict: Transcription data.
    
    Raises:
        AuraVoxError: If transcription fails or polling errors.
    """
    try:
        while True:
            response = requests.get(
                f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            if data["status"] == "completed":
                return data
            elif data["status"] == "failed":
                raise AuraVoxError("Transcription failed.")
            time.sleep(5)
    except Exception as e:
        raise AuraVoxError(f"Transcription retrieval failed: {e}")

def diarize_audio_assemblyai(audio_url):
    """
    Perform speaker diarization using AssemblyAI API.
    
    Args:
        audio_url (str): URL of the uploaded audio.
    
    Returns:
        str: Diarization transcript ID.
    
    Raises:
        AuraVoxError: If diarization request fails.
    """
    try:
        payload = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "language_detection": True
        }
        response = requests.post(
            f"{ASSEMBLYAI_API_URL}/transcript",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["id"]
    except Exception as e:
        raise AuraVoxError(f"Diarization request failed: {e}")

def get_diarization_result(transcript_id):
    """
    Retrieve speaker-wise diarization result from AssemblyAI.
    
    Args:
        transcript_id (str): ID of the diarization transcript.
    
    Returns:
        list: List of utterances.
    
    Raises:
        AuraVoxError: If diarization fails or polling errors.
    """
    try:
        while True:
            response = requests.get(
                f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            if data["status"] == "completed":
                return data.get("utterances", [])
            elif data["status"] == "failed":
                raise AuraVoxError("Diarization failed.")
            time.sleep(5)
    except Exception as e:
        raise AuraVoxError(f"Diarization retrieval failed: {e}")

@lru_cache(maxsize=10)
def summarize_text_gemini(text):
    """
    Summarize text using Google Gemini API with structured output.
    
    Args:
        text (str): Text to summarize.
    
    Returns:
        str: Structured summary.
    
    Raises:
        AuraVoxError: If summary generation fails.
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            "Summarize the following transcript in three distinct sections: "
            "1. Summary: Provide a concise overview of the main topics discussed. "
            "2. Trendy Content: Highlight any viral or engaging moments suitable for sharing. "
            "3. Key Moments: List critical points or decisions made during the conversation. "
            "Use clear headings for each section (## Summary, ## Trendy Content, ## Key Moments) "
            "and ensure each section is populated with relevant content. If a section is not applicable, "
            "state 'No relevant content identified.'\n\n"
            f"Transcript:\n{text}"
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise AuraVoxError(f"Summary generation failed: {e}")

def chat_with_gemini(question, context, history=None):
    """
    Generate a conversational response with memory using Gemini.
    
    Args:
        question (str): User's question.
        context (dict): Transcription and summary context.
        history (list, optional): Chat history.
    
    Returns:
        str: Response text.
    
    Raises:
        AuraVoxError: If chat response generation fails.
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        history_text = ""
        if history:
            for turn in history:
                history_text += f"User: {turn['question']['content']}\nAssistant: {turn['answer']['content']}\n"

        # Check for trendy/catchy moment queries
        trendy_keywords = ["catchy", "trendy", "viral", "engaging", "popular", "highlight"]
        is_trendy_query = any(keyword in question.lower() for keyword in trendy_keywords)

        if is_trendy_query:
            # Extract trendy items from summary
            summary_text = context['summary']
            trendy_match = re.search(r'##\s*Trendy Content\s*(.*?)(##\s*Key Moments|$)', summary_text, re.DOTALL | re.IGNORECASE)
            trendy_items = []
            if trendy_match:
                trendy_content = trendy_match.group(1).strip()
                raw_lines = [ln.strip() for ln in trendy_content.split("\n")]
                trendy_items = [re.sub(r"^(\-|\|•|\d+\.)\s", "", ln).strip() for ln in raw_lines if ln.strip()]

            if not trendy_items:
                response = "No trendy or catchy moments identified in the audio."
            else:
                response = "Here are the trendy or catchy moments from the audio:\n\n"
                diarization_data = st.session_state.get('diarization_data', [])
                audio_bytes = st.session_state.get('audio_bytes', None)
                mode = st.session_state.get('mode', 'loose')
                allow_bracket = st.session_state.get('allow_bracket', True)
                clip_format = st.session_state.get('clip_format', 'mp3')
                clip_padding = st.session_state.get('clip_padding', 150)
                clip_fade = st.session_state.get('clip_fade', 50)
                
                for idx, item in enumerate(trendy_items, 1):
                    response += f"*Trendy Moment {idx}*: {item}\n"
                    span = find_trendy_span_ms(diarization_data, item, mode=mode, allow_bracket_fallback=allow_bracket)
                    if span:
                        start_ms, end_ms = span
                        response += f"- *Timestamps*: {ms_to_mmss(start_ms)} – {ms_to_mmss(end_ms)}\n"
                        if PYDUB_AVAILABLE and audio_bytes:
                            try:
                                clip_bytes = make_clip_bytes(audio_bytes, start_ms, end_ms, out_format=clip_format, padding_ms=clip_padding, fade_ms=clip_fade)
                                if clip_bytes:
                                    clip_id = f"trendy_clip_{idx}_{ms_to_mmss(start_ms)}-{ms_to_mmss(end_ms)}"
                                    st.audio(BytesIO(clip_bytes), format=f"audio/{clip_format}")
                                    st.download_button(
                                        label=f"Download Clip {idx}",
                                        data=clip_bytes,
                                        file_name=f"{clip_id}.{clip_format}",
                                        mime=f"audio/{clip_format}",
                                        key=f"chat_clip_{idx}"
                                    )
                            except Exception as e:
                                response += f"- *Note*: Could not generate clip ({e}).\n"
                        else:
                            response += "- *Note*: Audio clipping unavailable (check pydub/ffmpeg or audio data).\n"
                    else:
                        response += "- *Note*: Could not locate timestamps for this moment.\n"
                    response += "\n"
        else:
            # General question handling
            prompt = (
                "You are a helpful chatbot answering questions based on the following audio transcription and summary. "
                "Provide accurate and concise answers relevant to the provided context. If the question is unrelated, "
                "inform the user politely.\n\n"
                f"Transcription:\n{context['transcription']}\n\n"
                f"Summary:\n{context['summary']}\n\n"
                f"Conversation History:\n{history_text}\n"
                f"User Question: {question}"
            )
            response = model.generate_content(prompt).text

        return response
    except Exception as e:
        raise AuraVoxError(f"Chat response generation failed: {e}")

# Helpers for trendy timestamps & clips
@lru_cache(maxsize=100)
def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9']+", s))

def _similarity_score(query_norm: str, window_norm: str) -> float:
    q_tokens = _token_set(query_norm)
    w_tokens = _token_set(window_norm)
    if not q_tokens or not w_tokens:
        jaccard = 0.0
    else:
        inter = len(q_tokens & w_tokens)
        union = len(q_tokens | w_tokens)
        jaccard = inter / max(1, union)
    ratio = SequenceMatcher(None, query_norm, window_norm).ratio()
    return 0.65 * jaccard + 0.35 * ratio

def _parse_bracket_timestamp(item_text):
    """
    Parse various timestamp formats:
    - [A 13:55 - 14:25]
    - [13:55 - 14:25]
    - (13:55 - 14:25)
    - 13:55-14:25
    - 13m55s-14m25s
    """
    patterns = [
        r"\[?\(?[A-Za-z]?\s*([0-9]{1,2}(?::[0-9]{2}){1,2})\s*[-–—]\s*([0-9]{1,2}(?::[0-9]{2}){1,2})\s*\)?\]?",  # [A 13:55 - 14:25]
        r"([0-9]{1,2}(?::[0-9]{2}){1,2})\s*[-–—]\s*([0-9]{1,2}(?::[0-9]{2}){1,2})",  # 13:55-14:25
        r"([0-9]{1,2}m[0-9]{2}s)\s*[-–—]\s*([0-9]{1,2}m[0-9]{2}s)"  # 13m55s-14m25s
    ]
    for pattern in patterns:
        m = re.search(pattern, item_text)
        if m:
            def to_ms(t):
                if "m" in t and "s" in t:
                    mm, ss = re.match(r"(\d+)m(\d+)s", t).groups()
                    return (int(mm) * 60 + int(ss)) * 1000
                parts = [int(x) for x in t.split(":")]
                if len(parts) == 3:
                    h, mm, ss = parts
                elif len(parts) == 2:
                    h = 0
                    mm, ss = parts
                else:
                    return None
                return ((h * 3600) + (mm * 60) + ss) * 1000
            try:
                start_ms = to_ms(m.group(1))
                end_ms = to_ms(m.group(2))
                if start_ms is not None and end_ms is not None and end_ms > start_ms:
                    return int(start_ms), int(end_ms)
            except Exception:
                continue
    return None

def find_trendy_span_ms(diarization_data, item_text, mode="loose", allow_bracket_fallback=True):
    """
    Find a clip span for item_text using multiple strategies:
    1) Bracketed timestamps
    2) Text similarity (Jaccard + SequenceMatcher)
    3) Semantic similarity (if available)
    4) Loose fallback to best utterance
    """
    if not diarization_data or not item_text or not item_text.strip():
        return None

    # 1. Try bracketed timestamps
    if allow_bracket_fallback:
        br = _parse_bracket_timestamp(item_text)
        if br:
            return br

    item_norm = _normalize_text(item_text)
    if not item_norm:
        return None

    MAX_WINDOW_UTTS = 8
    MIN_SCORE = 0.28 if mode == "loose" else 0.45
    MIN_CLIP_MS = 2000
    MAX_CLIP_MS = 25000

    n = len(diarization_data)
    norm_texts = [_normalize_text(seg.get("text", "")) for seg in diarization_data]

    # 2. Text similarity (Jaccard + SequenceMatcher)
    best_text = None
    for i in range(n):
        if not norm_texts[i]:
            continue
        s = diarization_data[i].get("start", 0) or 0
        end_time = s
        concat = []
        for j in range(i, min(n, i + MAX_WINDOW_UTTS)):
            seg = diarization_data[j]
            seg_norm = norm_texts[j]
            if seg_norm:
                concat.append(seg_norm)
            end_time = seg.get("end", end_time) or end_time
            duration = end_time - s
            if duration > 30_000:
                break
            window_text = " ".join(concat)
            jaccard_score = _similarity_score(item_norm, window_text)
            # Dynamic weighting based on item length
            item_len = len(item_norm.split())
            jaccard_weight = 0.8 if item_len < 5 else 0.6
            seq_weight = 1.0 - jaccard_weight
            score = jaccard_weight * jaccard_score + seq_weight * SequenceMatcher(None, item_norm, window_text).ratio()
            if best_text is None or score > best_text[0]:
                best_text = (score, int(s), int(end_time))

    # 3. Semantic similarity (if available)
    best_semantic = None
    if EMBEDDINGS_AVAILABLE:
        item_embedding = EMBEDDING_MODEL.encode(item_text, convert_to_tensor=True)
        for i in range(n):
            if not norm_texts[i]:
                continue
            s = diarization_data[i].get("start", 0) or 0
            end_time = diarization_data[i].get("end", s) or s
            seg_text = diarization_data[i].get("text", "")
            if seg_text:
                seg_embedding = EMBEDDING_MODEL.encode(seg_text, convert_to_tensor=True)
                score = util.cos_sim(item_embedding, seg_embedding).item()
                if best_semantic is None or score > best_semantic[0]:
                    best_semantic = (score, int(s), int(end_time))

    # Choose best match
    final_best = None
    if best_text and best_text[0] >= MIN_SCORE:
        final_best = best_text
    elif best_semantic and best_semantic[0] >= 0.7:  # Threshold for semantic similarity
        final_best = best_semantic

    if final_best:
        score, s, e = final_best
        dur = e - s
        if dur < MIN_CLIP_MS:
            e = s + MIN_CLIP_MS
        if e - s > MAX_CLIP_MS:
            mid = s + dur // 2
            s = max(0, mid - MAX_CLIP_MS // 2)
            e = s + MAX_CLIP_MS
        return int(s), int(e)

    # 4. Fallback to fragments or loose mode
    fragments = [f.strip() for f in re.split(r"[.,;:!?]\s*", item_text) if f.strip()]
    for frag in fragments:
        frag_norm = _normalize_text(frag)
        if not frag_norm:
            continue
        best_frag = None
        for i in range(n):
            if not norm_texts[i]:
                continue
            s = diarization_data[i].get("start", 0) or 0
            end_time = diarization_data[i].get("end", s) or s
            window_text = norm_texts[i]
            score = _similarity_score(frag_norm, window_text)
            if best_frag is None or score > best_frag[0]:
                best_frag = (score, int(s), int(end_time))
        if best_frag and best_frag[0] >= (MIN_SCORE * 0.9):
            s, e = best_frag[1], best_frag[2]
            if e - s < MIN_CLIP_MS:
                e = s + MIN_CLIP_MS
            return int(s), int(e)

    if mode == "loose" and best_text and best_text[0] > 0.12:
        s, e = best_text[1], best_text[2]
        if e - s < MIN_CLIP_MS:
            e = s + MIN_CLIP_MS
        return int(s), int(e)

    return None

def ms_to_mmss(ms: int) -> str:
    s = max(0, int(ms // 1000))
    return f"{s//60:02d}:{s%60:02d}"

def make_clip_bytes(audio_bytes: bytes, start_ms: int, end_ms: int, out_format="mp3", padding_ms=150, fade_ms=50) -> bytes:
    """
    Generate an audio clip from the given bytes with padding and fade effects.
    
    Args:
        audio_bytes (bytes): Original audio bytes.
        start_ms (int): Start time in milliseconds.
        end_ms (int): End time in milliseconds.
        out_format (str, optional): Output format (mp3 or wav). Defaults to "mp3".
        padding_ms (int, optional): Padding in milliseconds. Defaults to 150.
        fade_ms (int, optional): Fade in/out duration in milliseconds. Defaults to 50.
    
    Returns:
        bytes: Clipped audio bytes.
    """
    if not PYDUB_AVAILABLE:
        st.warning("pydub is not available. Install pydub and ffmpeg to enable audio clipping.")
        return b""
    
    try:
        # Validate audio file
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        if not audio.frame_rate or audio.frame_count() == 0:
            st.error("Invalid audio file: No audio data detected.")
            return b""
        
        total = len(audio)
        s = max(0, min(start_ms, total))
        e = max(s, min(end_ms, total))
        if e <= s:
            e = min(total, s + 2000)
        
        # Apply padding
        s = max(0, s - padding_ms)
        e = min(total, e + padding_ms)
        
        clip = audio[s:e]
        
        # Apply fade effects if specified
        if fade_ms > 0:
            try:
                clip = clip.fade_in(fade_ms).fade_out(fade_ms)
            except Exception as e:
                st.warning(f"Failed to apply fade effects: {e}")
        
        # Export with quality settings
        buf = BytesIO()
        if out_format == "mp3":
            clip.export(buf, format="mp3", bitrate="192k")
        elif out_format == "wav":
            clip.export(buf, format="wav")
        else:
            st.warning(f"Unsupported output format: {out_format}. Defaulting to mp3.")
            clip.export(buf, format="mp3", bitrate="192k")
        
        return buf.getvalue()
    except Exception as e:
        st.error(f"Audio clipping failed: {e}")
        return b""

# Streamlit App
st.set_page_config(
    page_title="AURA VOX - Audio AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar options
with st.sidebar:
    st.markdown("## ⚙️ Matching Options")
    mode = st.radio("Matching mode", options=["loose", "strict"], index=0, help="loose = tolerant matching + fallbacks; strict = high precision")
    st.session_state['mode'] = mode
    allow_bracket = st.checkbox("Allow bracket-timestamp fallback (use provided [A 13:55 - 14:25] if present)", value=True)
    st.session_state['allow_bracket'] = allow_bracket
    st.divider()
    
    st.markdown("## 🎵 Audio Clipping Options")
    clip_format = st.selectbox("Output format", ["mp3", "wav"], index=0)
    st.session_state['clip_format'] = clip_format
    clip_padding = st.slider("Padding (ms)", 0, 500, 150, step=50)
    st.session_state['clip_padding'] = clip_padding
    clip_fade = st.slider("Fade in/out (ms)", 0, 200, 50, step=10)
    st.session_state['clip_fade'] = clip_fade
    st.divider()

    st.markdown("Upload an audio file (mp3, wav, m4a, ogg) and click Process Audio.")
    if not PYDUB_AVAILABLE:
        st.warning("Audio clipping requires pydub + ffmpeg. Please install them to enable clip downloads.")

# Top header
with st.container():
    hero_html = """
    <div class="hero-box">
      <div class="hero-title">✨ AURA VOX</div>
      <div class="hero-sub">🎙️ Audio Transcription, Diarization & AI Insights</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

left_col, right_col = st.columns([1,2], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an audio file to upload", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file is not None:
        st.success(f"File ready: {uploaded_file.name}")
        audio_data = uploaded_file.read()
    else:
        audio_data = None

    if uploaded_file is not None:
        if st.button("Process Audio", key="process_btn"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            # Step 1: Upload
            status_placeholder.info("Uploading audio to AssemblyAI...")
            progress_bar.progress(0.2)
            try:
                audio_url = upload_audio(audio_data, uploaded_file.name)
                status_placeholder.success("Audio uploaded successfully.")
            except AuraVoxError as e:
                status_placeholder.error(str(e))
                progress_bar.progress(0.0)
                st.stop()

            # Step 2: Transcription
            status_placeholder.info("Transcribing audio...")
            progress_bar.progress(0.4)
            try:
                transcript_id = transcribe_audio_assemblyai(audio_url)
                transcript_data = get_transcription_result(transcript_id)
                status_placeholder.success("Transcription completed.")
            except AuraVoxError as e:
                status_placeholder.error(str(e))
                progress_bar.progress(0.0)
                st.stop()

            # Step 3: Diarization
            status_placeholder.info("Performing speaker diarization...")
            progress_bar.progress(0.6)
            try:
                diarization_id = diarize_audio_assemblyai(audio_url)
                diarization_data = get_diarization_result(diarization_id)
                status_placeholder.success("Diarization completed.")
            except AuraVoxError as e:
                status_placeholder.error(str(e))
                progress_bar.progress(0.0)
                st.stop()

            aligned_dialogue = []
            for segment in diarization_data:
                speaker = segment.get("speaker", "Unknown")
                start_time = segment.get("start", 0) / 1000
                end_time = segment.get("end", 0) / 1000
                text = segment.get("text", "")
                aligned_dialogue.append(f"[{speaker} {start_time:.2f}s - {end_time:.2f}s] {text}")

            # Step 4: Summarization
            status_placeholder.info("Generating summary with Gemini...")
            progress_bar.progress(0.8)
            try:
                dialogue_text = "\n".join(aligned_dialogue)
                summary = summarize_text_gemini(dialogue_text)
                status_placeholder.success("Summary generated.")
            except AuraVoxError as e:
                status_placeholder.error(str(e))
                progress_bar.progress(0.0)
                st.stop()

            # Extract trendy items for DB save
            trendy_clips = []
            trendy_match = re.search(r'##\s*Trendy Content\s*(.*?)(##\s*Key Moments|$)', summary, re.DOTALL | re.IGNORECASE)
            if trendy_match:
                trendy_content = trendy_match.group(1).strip()
                raw_lines = [ln.strip() for ln in trendy_content.split("\n")]
                trendy_items = [re.sub(r"^(\-|\|•|\d+\.)\s", "", ln).strip() for ln in raw_lines if ln.strip()]
                if trendy_items and PYDUB_AVAILABLE and audio_data:
                    for item in trendy_items:
                        span = find_trendy_span_ms(diarization_data, item, mode=mode, allow_bracket_fallback=allow_bracket)
                        if span:
                            start_ms, end_ms = span
                            clip_bytes = make_clip_bytes(audio_data, start_ms, end_ms, out_format=clip_format, padding_ms=clip_padding, fade_ms=clip_fade)
                            trendy_clips.append((item, start_ms, end_ms, clip_bytes))

            # Step 5: Save to DB
            progress_bar.progress(0.9)
            try:
                save_to_db(
                    uploaded_file.name,
                    summary,
                    "\n".join(aligned_dialogue),
                    diarization_data,
                    trendy_clips
                )
                status_placeholder.success("✅ Data saved to database.")
                progress_bar.progress(1.0)
            except AuraVoxError as e:
                status_placeholder.error(str(e))
                progress_bar.progress(0.0)
                st.stop()

            st.session_state['aligned_dialogue'] = aligned_dialogue
            st.session_state['summary'] = summary
            st.session_state['diarization_data'] = diarization_data
            st.session_state['audio_bytes'] = audio_data
                
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dialogue", "Summary", "Trendy Content", "Key Moments", "Chat"])

    with tab1:
        st.header("🗣️ Speaker-Labeled Dialogue")
        if 'aligned_dialogue' in st.session_state:
            for line in st.session_state['aligned_dialogue']:
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    timestamp_label = parts[0][1:]
                    text = parts[1]
                    st.markdown(f"<div class=\"dialogue-box\"><div class=\"speaker-label\">{timestamp_label}</div><div style=\"margin-top:8px\">{text}</div></div>", unsafe_allow_html=True)
            st.download_button(label="Download Dialogue", data="\n".join(st.session_state['aligned_dialogue']), file_name="aligned_dialogue.txt", mime="text/plain")
        else:
            st.info("Process an audio file to view dialogue.")

    with tab2:
        st.header("📝 Summary")
        if 'summary' in st.session_state:
            summary_text = st.session_state['summary']
            summary_section = re.split(r'##\s*Trendy Content', summary_text, 1)[0].strip()
            st.markdown(summary_section)
            st.download_button(label="Download Summary", data=summary_text, file_name="summary.txt", mime="text/plain")
        else:
            st.info("Process an audio file to view summary.")

    with tab3:
        st.header("🔥 Trendy Content")
        if 'summary' in st.session_state:
            summary_text = st.session_state['summary']
            trendy_match = re.search(r'##\s*Trendy Content\s*(.*?)(##\s*Key Moments|$)', summary_text, re.DOTALL | re.IGNORECASE)
            if trendy_match:
                trendy_content = trendy_match.group(1).strip()
                raw_lines = [ln.strip() for ln in trendy_content.split("\n")]
                trendy_items = [re.sub(r"^(\-|\|•|\d+\.)\s", "", ln).strip() for ln in raw_lines if ln.strip()]

                if not trendy_items:
                    st.info("No trendy content identified.")
                else:
                    for idx, item in enumerate(trendy_items, start=1):
                        with st.expander(f"✨ Trendy Clip {idx}: {item[:50]}...", expanded=False):
                            st.markdown(f"*Text:* {item}")
                            span = find_trendy_span_ms(st.session_state.get('diarization_data', []), item, mode=mode, allow_bracket_fallback=allow_bracket)
                            if span:
                                start_ms, end_ms = span
                                st.markdown(f"*Timestamps:* {ms_to_mmss(start_ms)} – {ms_to_mmss(end_ms)}")
                                if PYDUB_AVAILABLE and 'audio_bytes' in st.session_state:
                                    try:
                                        clip_bytes = make_clip_bytes(
                                            st.session_state['audio_bytes'],
                                            start_ms,
                                            end_ms,
                                            out_format=clip_format,
                                            padding_ms=clip_padding,
                                            fade_ms=clip_fade
                                        )
                                        if clip_bytes:
                                            st.audio(BytesIO(clip_bytes), format=f"audio/{clip_format}")
                                            st.download_button(
                                                label=f"⬇️ Download Clip {idx}",
                                                data=clip_bytes,
                                                file_name=f"trendy_clip_{idx}_{ms_to_mmss(start_ms)}-{ms_to_mmss(end_ms)}.{clip_format}",
                                                mime=f"audio/{clip_format}",
                                                key=f"dl_trendy_{idx}"
                                            )
                                        else:
                                            st.warning("Clip could not be generated.")
                                    except Exception as e:
                                        st.warning(f"Clip error: {e}")
                                else:
                                    st.info("Audio clipping unavailable (check pydub/ffmpeg or audio data).")
                            else:
                                st.warning("Could not locate timestamps for this item.")
            else:
                st.info("No trendy content identified.")
        else:
            st.info("Process an audio file to view trendy content.")

    with tab4:
        st.header("⭐ Key Moments")
        if 'summary' in st.session_state:
            summary_text = st.session_state['summary']
            key_moments_match = re.search(r'##\s*Key Moments\s*(.*)', summary_text, re.DOTALL | re.IGNORECASE)
            if key_moments_match:
                key_moments = key_moments_match.group(1).strip()
                st.markdown(key_moments)
            else:
                st.info("No key moments identified.")
        else:
            st.info("Process an audio file to view key moments.")

    with tab5:
        st.header("💬 Chat About the Audio")
        if 'aligned_dialogue' in st.session_state and 'summary' in st.session_state:
            # Preview original audio
            if 'audio_bytes' in st.session_state:
                st.audio(BytesIO(st.session_state['audio_bytes']), format="audio/mp3" if uploaded_file.type == "audio/mpeg" else uploaded_file.type)
            
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []

            user_input = st.chat_input("Ask a question about the audio content (e.g., 'What are the trendy moments?')...")
            if user_input:
                context = {
                    "transcription": "\n".join(st.session_state['aligned_dialogue']),
                    "summary": st.session_state['summary']
                }
                try:
                    with st.spinner("Generating chat response..."):
                        response = chat_with_gemini(user_input, context, st.session_state['chat_history'])
                    st.session_state['chat_history'].append({
                        "question": {"role": "user", "content": user_input},
                        "answer": {"role": "assistant", "content": response}
                    })
                except AuraVoxError as e:
                    st.error(str(e))

            if st.button("Clear Chat History", key="clear_chat_btn"):
                st.session_state['chat_history'] = []

            for pair in st.session_state['chat_history']:
                with st.chat_message(pair["question"]["role"]):
                    st.markdown(pair["question"]["content"])
                with st.chat_message(pair["answer"]["role"]):
                    st.markdown(pair["answer"]["content"])
        else:
            st.info("Process an audio file to enable the chat feature.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ — AssemblyAI + Gemini. Keep API keys private in production.")

with st.expander("📂 View Saved Transcripts"):
    transcripts = fetch_all_transcripts()
    if transcripts:
        for tid, fname, summary, created in transcripts:
            st.markdown(f"*ID {tid}* — {fname} ({created})")
            st.text_area("Summary", summary, height=100, key=f"summary_{tid}")
            st.divider()
    else:
        st.info("No transcripts saved yet.")

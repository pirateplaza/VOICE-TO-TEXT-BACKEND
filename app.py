from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
import os
import tempfile
import json
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sqlite3

# =============================================
# AUTO-DOWNLOAD REQUIRED NLTK DATA
# =============================================
def download_nltk_data():
    packages = ['punkt', 'stopwords', 'punkt_tab', 'averaged_perceptron_tagger']
    for pkg in packages:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except:
                pass
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

download_nltk_data()

# =============================================
# FLASK APP SETUP
# =============================================
app = Flask(__name__)
CORS(app)

# =============================================
# DATABASE SETUP (SQLite)
# =============================================
def init_db():
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  transcript TEXT,
                  summary TEXT,
                  keywords TEXT,
                  quiz TEXT,
                  created_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

# =============================================
# ML FUNCTIONS
# =============================================

def extract_keywords(text, n=8):
    """Extract top keywords using TF-IDF (Machine Learning method)"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=n)
        vectorizer.fit([text])
        return list(vectorizer.get_feature_names_out())
    except Exception:
        # Fallback: manual keyword extraction using NLTK
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
            freq = {}
            for w in filtered:
                freq[w] = freq.get(w, 0) + 1
            sorted_words = sorted(freq, key=freq.get, reverse=True)
            return sorted_words[:n]
        except:
            return []


def generate_summary(text, num_sentences=3):
    """Generate extractive summary using TF-IDF scoring"""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices = scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except Exception:
        # Fallback: return first 3 sentences
        sentences = text.split('. ')
        return '. '.join(sentences[:num_sentences]) + '.'


def generate_quiz(text, n=5):
    """Auto-generate fill-in-the-blank quiz questions using NLTK"""
    try:
        sentences = sent_tokenize(text)
        questions = []

        for sentence in sentences:
            if len(questions) >= n:
                break

            words = word_tokenize(sentence)
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()

            # Find important words to blank out
            meaningful = [
                (i, w) for i, w in enumerate(words)
                if w.isalpha() and w.lower() not in stop_words and len(w) > 3
            ]

            if len(meaningful) >= 2 and len(words) > 5:
                idx, word = meaningful[len(meaningful) // 2]
                blanked = words.copy()
                blanked[idx] = '_____'
                question = ' '.join(blanked)
                questions.append({
                    'question': question,
                    'answer': word,
                    'original': sentence
                })

        return questions
    except Exception:
        return []


# =============================================
# API ROUTES
# =============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Voice Learning App Backend is Running!',
        'status': 'OK',
        'endpoints': ['/transcribe', '/save', '/sessions']
    })


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Receive audio file, convert speech to text, run ML analysis"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file received'}), 400

    audio_file = request.files['audio']
    recognizer = sr.Recognizer()

    # Browser sends real PCM WAV (converted by App.js before upload)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        # Using Google Speech Recognition (free, no API key needed)
        text = recognizer.recognize_google(audio_data)

        if not text or len(text.strip()) < 5:
            return jsonify({'error': 'Speech was too short. Please speak more.'}), 400

        # ML PROCESSING
        summary = generate_summary(text)
        keywords = extract_keywords(text)
        quiz = generate_quiz(text)

        return jsonify({
            'text': text,
            'summary': summary,
            'keywords': keywords,
            'quiz': quiz,
            'word_count': len(text.split())
        })

    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand the audio. Please speak clearly and try again.'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Internet connection needed for speech recognition. Error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.route('/save', methods=['POST'])
def save_session():
    """Save a learning session to the database"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400

    title = data.get('title', 'Session ' + datetime.now().strftime('%d %b %H:%M'))
    transcript = data.get('transcript', '')
    summary = data.get('summary', '')
    keywords = json.dumps(data.get('keywords', []))
    quiz = json.dumps(data.get('quiz', []))
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO sessions (title, transcript, summary, keywords, quiz, created_at) VALUES (?,?,?,?,?,?)',
        (title, transcript, summary, keywords, quiz, created_at)
    )
    conn.commit()
    session_id = c.lastrowid
    conn.close()

    return jsonify({'id': session_id, 'message': 'Session saved successfully!'})


@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Get all saved sessions"""
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    c.execute('SELECT id, title, created_at FROM sessions ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return jsonify([{'id': r[0], 'title': r[1], 'created_at': r[2]} for r in rows])


@app.route('/sessions/<int:session_id>', methods=['GET'])
def get_session(session_id):
    """Get a specific session by ID"""
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return jsonify({
            'id': row[0],
            'title': row[1],
            'transcript': row[2],
            'summary': row[3],
            'keywords': json.loads(row[4]) if row[4] else [],
            'quiz': json.loads(row[5]) if row[5] else [],
            'created_at': row[6]
        })
    return jsonify({'error': 'Session not found'}), 404


@app.route('/sessions/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Session deleted successfully'})


# =============================================
# RUN THE APP
# =============================================
if __name__ == '__main__':
    print("=" * 50)
    print("  Voice Learning App - Backend Started!")
    print("  API running at: http://localhost:5000")
    print("  Press CTRL+C to stop the server")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')

import os
import io
import json
import time
import zipfile
import threading

import requests as http_requests
from flask import Flask, request, render_template_string, send_file
from openai import AzureOpenAI


# ---------------------------------------------------------------------------
# Local dev only: load vars.env.txt if it exists (ignored on Azure)
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(__file__), "vars.env.txt")
if os.path.isfile(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _raw in _f:
            _line = _raw.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k:
                os.environ[_k] = _v

AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", "")

ai_client = None
if AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT:
    ai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-10-21",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Persistent visitor counter (thread-safe, file-backed)
# ---------------------------------------------------------------------------
_counter_path = os.path.join(os.path.dirname(__file__), "visitor_count.json")
_counter_lock = threading.Lock()


def _read_counter():
    try:
        with open(_counter_path, "r") as f:
            return json.load(f).get("count", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _increment_counter():
    with _counter_lock:
        count = _read_counter() + 1
        with open(_counter_path, "w") as f:
            json.dump({"count": count}, f)
        return count


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CuratorAI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      color: #e0e0e0;
      min-height: 100vh;
      padding: 32px 24px;
    }

    .container { max-width: 960px; margin: 0 auto; }

    /* Header */
    .header { text-align: center; margin-bottom: 40px; }
    .header h1 {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    .header p { color: #9ca3af; font-size: 0.95rem; }

    /* Cards */
    .card {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 32px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      margin-bottom: 24px;
    }

    /* Form elements */
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 8px;
      font-size: 0.9rem;
      color: #c4b5fd;
    }
    textarea, input[type="number"] {
      width: 100%;
      padding: 14px 18px;
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 12px;
      background: rgba(255,255,255,0.07);
      color: #f0f0f0;
      font-size: 1rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus, input[type="number"]:focus {
      border-color: #a78bfa;
      box-shadow: 0 0 0 3px rgba(167,139,250,0.25);
    }
    textarea { resize: vertical; min-height: 100px; }
    textarea::placeholder { color: #6b7280; }
    input[type="number"] { max-width: 160px; }

    .form-row { margin-bottom: 20px; }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 14px 32px;
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.2s;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }

    .btn-primary {
      background: linear-gradient(135deg, #7c3aed, #6366f1);
      color: #fff;
    }
    .btn-primary:hover { box-shadow: 0 4px 16px rgba(99,102,241,0.45); }

    .btn-success {
      background: linear-gradient(135deg, #059669, #10b981);
      color: #fff;
    }
    .btn-success:hover { box-shadow: 0 4px 16px rgba(16,185,129,0.45); }

    /* Error / info */
    .error-box {
      background: rgba(239,68,68,0.15);
      border: 1px solid rgba(239,68,68,0.3);
      color: #fca5a5;
      padding: 14px 20px;
      border-radius: 12px;
      margin-bottom: 20px;
      font-size: 0.9rem;
      line-height: 1.6;
    }

    /* Image grid */
    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 4px;
    }
    .results-count { color: #9ca3af; font-size: 0.85rem; margin-top: 2px; }

    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .image-grid img {
      width: 100%;
      height: 150px;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.1);
      transition: transform 0.2s, border-color 0.2s;
    }
    .image-grid img:hover {
      transform: scale(1.03);
      border-color: rgba(167,139,250,0.5);
    }

    /* Queries list */
    .queries-used {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid rgba(255,255,255,0.08);
    }
    .queries-used summary {
      cursor: pointer;
      font-size: 0.8rem;
      color: #6b7280;
    }
    .queries-used ul {
      margin-top: 8px;
      padding-left: 20px;
      font-size: 0.8rem;
      color: #9ca3af;
    }

    .empty-state {
      text-align: center;
      color: #6b7280;
      font-style: italic;
      padding: 40px 0;
    }

    /* Visitor counter */
    .visitor-counter {
      position: fixed;
      top: 20px;
      left: 20px;
      z-index: 100;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
      padding: 8px 14px;
      backdrop-filter: blur(8px);
      font-size: 0.75rem;
      color: #9ca3af;
    }
    .visitor-counter span { color: #a78bfa; font-weight: 600; }

    /* Image card with checkbox */
    .image-card {
      position: relative;
      cursor: pointer;
    }
    .image-card input[type="checkbox"] {
      position: absolute;
      top: 8px;
      right: 8px;
      width: 22px;
      height: 22px;
      accent-color: #7c3aed;
      cursor: pointer;
      z-index: 2;
    }
    .image-card.deselected img {
      opacity: 0.35;
      filter: grayscale(0.6);
    }

    .selection-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
      padding: 10px 16px;
      background: rgba(167,139,250,0.1);
      border: 1px solid rgba(167,139,250,0.2);
      border-radius: 10px;
      font-size: 0.85rem;
      color: #c4b5fd;
    }
    .selection-bar .count { font-weight: 700; }
    .selection-bar button {
      background: none;
      border: 1px solid rgba(167,139,250,0.3);
      color: #a78bfa;
      padding: 4px 12px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
      font-family: inherit;
      transition: background 0.2s;
    }
    .selection-bar button:hover { background: rgba(167,139,250,0.15); }

    .support {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 100;
    }
    .btn-kofi {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 20px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(135deg, #ff5e5b, #ff9966);
      color: #fff;
      font-size: 0.85rem;
      font-weight: 600;
      font-family: inherit;
      text-decoration: none;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.2s;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .btn-kofi:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(255, 94, 91, 0.45);
    }
    .btn-kofi:active { transform: translateY(0); }

    .nav {
      display: flex;
      justify-content: center;
      gap: 24px;
      margin-bottom: 28px;
    }
    .nav a {
      color: #9ca3af;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      padding: 6px 0;
      border-bottom: 2px solid transparent;
      transition: color 0.2s, border-color 0.2s;
    }
    .nav a:hover, .nav a.active {
      color: #c4b5fd;
      border-bottom-color: #a78bfa;
    }

    .footer {
      text-align: center;
      margin-top: 16px;
      font-size: 0.75rem;
      color: #4b5563;
    }

    /* Loading overlay */
    .loading-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 12, 41, 0.85);
      backdrop-filter: blur(6px);
      z-index: 9999;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 20px;
    }
    .loading-overlay.active { display: flex; }
    .spinner {
      width: 48px;
      height: 48px;
      border: 4px solid rgba(167, 139, 250, 0.2);
      border-top-color: #a78bfa;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-overlay p {
      color: #c4b5fd;
      font-size: 1.05rem;
      font-weight: 500;
    }
    .loading-overlay .subtext {
      color: #6b7280;
      font-size: 0.8rem;
      font-weight: 400;
    }

    /* Lightbox */
    .lightbox {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.88);
      backdrop-filter: blur(8px);
      z-index: 10000;
      align-items: center;
      justify-content: center;
      cursor: zoom-out;
    }
    .lightbox.active { display: flex; }
    .lightbox img {
      max-width: 90vw;
      max-height: 85vh;
      border-radius: 12px;
      box-shadow: 0 12px 48px rgba(0,0,0,0.6);
      object-fit: contain;
    }
    .lightbox-title {
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      color: #d1d5db;
      font-size: 0.85rem;
      background: rgba(0,0,0,0.5);
      padding: 6px 16px;
      border-radius: 8px;
      max-width: 80vw;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .lightbox-close {
      position: absolute;
      top: 20px;
      right: 24px;
      background: none;
      border: none;
      color: #9ca3af;
      font-size: 2rem;
      cursor: pointer;
      line-height: 1;
      transition: color 0.2s;
    }
    .lightbox-close:hover { color: #fff; }
  </style>
</head>
<body>
  <div class="loading-overlay" id="loadingOverlay">
    <div class="spinner"></div>
    <p id="loadingText">Searching for images...</p>
    <span class="subtext">This may take a few seconds</span>
  </div>

  <div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
    <img id="lightboxImg" src="" alt="" />
    <div class="lightbox-title" id="lightboxTitle"></div>
  </div>

  <div class="visitor-counter">
    Visitors: <span>{{ visitor_count }}</span>
  </div>

  <div class="support">
    <a href="https://ko-fi.com/coolcoderme" target="_blank" rel="noopener" class="btn btn-kofi">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <path d="M23.881 8.948c-.773-4.085-4.859-4.593-4.859-4.593H.723c-.604 0-.679.798-.679.798s-.082 7.324-.022 11.822c.164 2.424 2.586 2.672 2.586 2.672s8.267-.023 11.966-.049c2.438-.426 2.683-2.566 2.658-3.734 4.352.24 7.422-2.831 6.649-6.916zm-11.062 3.511c-1.246 1.453-4.011 3.976-4.011 3.976s-.121.119-.31.023c-.076-.057-.108-.09-.108-.09-.443-.441-3.368-3.049-4.034-3.954-.709-.965-1.041-2.7-.091-3.71.951-1.01 3.005-1.086 4.363.407 0 0 1.565-1.782 3.468-.963 1.904.82 1.832 3.011.723 4.311zm6.173.478c-.928.116-1.682.028-1.682.028V7.284h1.77s1.971.551 1.971 2.638c0 1.913-.985 2.667-2.059 3.015z"/>
      </svg>
      Support
    </a>
  </div>
  <div class="container">
    <div class="header">
      <h1>CuratorAI</h1>
      <p>Describe what training images you need &mdash; AI finds Creative Commons pictures for you</p>
    </div>

    <div class="nav">
      <a href="/" class="{{ 'active' if active_page == 'home' }}">Home</a>
      <a href="/about" class="{{ 'active' if active_page == 'about' }}">About</a>
    </div>

    {% if error %}
    <div class="error-box">{{ error }}</div>
    {% endif %}

    <!-- Search form -->
    <div class="card">
      <form method="post" action="/">
        <div class="form-row">
          <label for="description">Describe the images you need</label>
          <textarea id="description" name="description"
            placeholder="e.g. Different breeds of dogs in outdoor settings for training a dog breed classifier">{{ description }}</textarea>
        </div>
        <div class="form-row">
          <label for="num_images">Number of images (5 &ndash; 50)</label>
          <input type="number" id="num_images" name="num_images"
                 value="{{ num_images }}" min="5" max="50" />
        </div>
        <button type="submit" class="btn btn-primary">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"
               viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Search Images
        </button>
      </form>
    </div>

    <!-- Results -->
    {% if images %}
    <div class="card">
      <div class="results-header">
        <div>
          <h2 style="font-size:1.2rem;">Found {{ images|length }} image{{ "s" if images|length != 1 }}</h2>
          <div class="results-count">Creative Commons licensed &middot; uncheck images you don't want, then download</div>
        </div>
        <form method="post" action="/download" id="downloadForm">
          <input type="hidden" name="urls" id="selectedUrls" value="{{ images_json }}" />
          <input type="hidden" name="description" value="{{ description }}" />
          <button type="submit" class="btn btn-success">
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"
                 viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                 <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Download ZIP
          </button>
        </form>
      </div>

      <div class="selection-bar">
        <div><span class="count" id="selectedCount">{{ images|length }}</span> of {{ images|length }} selected</div>
        <div>
          <button type="button" onclick="toggleAll(true)">Select all</button>
          <button type="button" onclick="toggleAll(false)">Deselect all</button>
        </div>
      </div>

      <div class="image-grid">
        {% for img in images %}
        <div class="image-card" data-url="{{ img.url }}">
          <input type="checkbox" checked onchange="updateSelection()" />
          <img src="{{ img.thumb }}" alt="{{ img.title }}" title="{{ img.title }}" loading="lazy"
               onclick="openLightbox('{{ img.url }}', '{{ img.title|e }}')" />
        </div>
        {% endfor %}
      </div>

      {% if queries_used %}
      <div class="queries-used">
        <details>
          <summary>Search queries used by AI</summary>
          <ul>
            {% for q in queries_used %}
            <li>{{ q }}</li>
            {% endfor %}
          </ul>
        </details>
      </div>
      {% endif %}
    </div>
    {% elif request_method == "POST" and not error %}
    <div class="card">
      <div class="empty-state">No Creative Commons images found. Try a different description.</div>
    </div>
    {% endif %}

    <div class="footer">Built with Flask, Azure OpenAI &amp; Openverse</div>
  </div>

  <script>
    function openLightbox(url, title) {
      var lb = document.getElementById('lightbox');
      document.getElementById('lightboxImg').src = url;
      document.getElementById('lightboxImg').alt = title;
      document.getElementById('lightboxTitle').textContent = title;
      lb.classList.add('active');
    }
    function closeLightbox() {
      document.getElementById('lightbox').classList.remove('active');
      document.getElementById('lightboxImg').src = '';
    }
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') closeLightbox();
    });

    function updateSelection() {
      var cards = document.querySelectorAll('.image-card');
      var urls = [];
      var total = cards.length;
      cards.forEach(function(card) {
        var cb = card.querySelector('input[type="checkbox"]');
        if (cb.checked) {
          urls.push(card.dataset.url);
          card.classList.remove('deselected');
        } else {
          card.classList.add('deselected');
        }
      });
      var countEl = document.getElementById('selectedCount');
      if (countEl) countEl.textContent = urls.length;
      var urlsInput = document.getElementById('selectedUrls');
      if (urlsInput) urlsInput.value = JSON.stringify(urls);
    }

    function toggleAll(checked) {
      document.querySelectorAll('.image-card input[type="checkbox"]').forEach(function(cb) {
        cb.checked = checked;
      });
      updateSelection();
    }

    document.querySelectorAll('form').forEach(function(form) {
      form.addEventListener('submit', function() {
        var overlay = document.getElementById('loadingOverlay');
        var text = document.getElementById('loadingText');
        if (form.action.includes('/download')) {
          text.textContent = 'Downloading and zipping images...';
          overlay.classList.add('active');
          setTimeout(function() {
            overlay.classList.remove('active');
          }, 3000);
        } else {
          text.textContent = 'Searching for images...';
          overlay.classList.add('active');
        }
      });
    });
    window.addEventListener('pageshow', function() {
      document.getElementById('loadingOverlay').classList.remove('active');
    });
  </script>
</body>
</html>
"""

ABOUT_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>About &mdash; CuratorAI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      color: #e0e0e0;
      min-height: 100vh;
      padding: 32px 24px;
    }
    .container { max-width: 960px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 40px; }
    .header h1 {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    .header p { color: #9ca3af; font-size: 0.95rem; }
    .nav {
      display: flex;
      justify-content: center;
      gap: 24px;
      margin-bottom: 28px;
    }
    .nav a {
      color: #9ca3af;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      padding: 6px 0;
      border-bottom: 2px solid transparent;
      transition: color 0.2s, border-color 0.2s;
    }
    .nav a:hover, .nav a.active {
      color: #c4b5fd;
      border-bottom-color: #a78bfa;
    }
    .card {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 32px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      margin-bottom: 24px;
    }
    .card h2 {
      font-size: 1.3rem;
      margin-bottom: 16px;
      color: #c4b5fd;
    }
    .card p {
      font-size: 0.95rem;
      line-height: 1.7;
      color: #d1d5db;
      margin-bottom: 12px;
    }
    .card ul {
      padding-left: 20px;
      margin-bottom: 12px;
    }
    .card li {
      font-size: 0.95rem;
      line-height: 1.7;
      color: #d1d5db;
      margin-bottom: 4px;
    }
    .card a { color: #a78bfa; }
    .card a:hover { color: #c4b5fd; }
    .support {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 100;
    }
    .btn-kofi {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 20px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(135deg, #ff5e5b, #ff9966);
      color: #fff;
      font-size: 0.85rem;
      font-weight: 600;
      font-family: inherit;
      text-decoration: none;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.2s;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .btn-kofi:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(255, 94, 91, 0.45);
    }
    .btn-kofi:active { transform: translateY(0); }
    .footer {
      text-align: center;
      margin-top: 32px;
      font-size: 0.75rem;
      color: #4b5563;
    }
  </style>
</head>
<body>
  <div class="support">
    <a href="https://ko-fi.com/coolcoderme" target="_blank" rel="noopener" class="btn btn-kofi">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <path d="M23.881 8.948c-.773-4.085-4.859-4.593-4.859-4.593H.723c-.604 0-.679.798-.679.798s-.082 7.324-.022 11.822c.164 2.424 2.586 2.672 2.586 2.672s8.267-.023 11.966-.049c2.438-.426 2.683-2.566 2.658-3.734 4.352.24 7.422-2.831 6.649-6.916zm-11.062 3.511c-1.246 1.453-4.011 3.976-4.011 3.976s-.121.119-.31.023c-.076-.057-.108-.09-.108-.09-.443-.441-3.368-3.049-4.034-3.954-.709-.965-1.041-2.7-.091-3.71.951-1.01 3.005-1.086 4.363.407 0 0 1.565-1.782 3.468-.963 1.904.82 1.832 3.011.723 4.311zm6.173.478c-.928.116-1.682.028-1.682.028V7.284h1.77s1.971.551 1.971 2.638c0 1.913-.985 2.667-2.059 3.015z"/>
      </svg>
      Support
    </a>
  </div>

  <div class="container">
    <div class="header">
      <h1>CuratorAI</h1>
      <p>Describe what training images you need &mdash; AI finds Creative Commons pictures for you</p>
    </div>

    <div class="nav">
      <a href="/">Home</a>
      <a href="/about" class="active">About</a>
    </div>

    <div class="card">
      <h2>About Me</h2>
      <p>Hi! I'm the creator of CuratorAI. (You can call me a coding geek and a nerdy teen)</p>
      <p>My sincere thanks to my parents for their support and encouragement while building this project. My sincere thanks also to my uncle, who understood my love of coding and gifted me robot kits, AI coding tools and so much more!</p>
    </div>

    <div class="card">
      <h2>About This Project</h2>
      <p>CuratorAI helps you collect Creative Commons licensed images for training AI models. Here's how it works:</p>
      <ul>
        <li>Describe the images you need in plain language</li>
        <li>AI generates smart, diverse search queries</li>
        <li>The app searches Openverse for Creative Commons images</li>
        <li>Preview the results and download them as a ZIP</li>
      </ul>
      <p>Built with Flask, Azure OpenAI, and Openverse.</p>
    </div>

    <div class="card">
      <h2>Support</h2>
      <p>If you find this tool useful, consider supporting the project on <a href="https://ko-fi.com/coolcoderme" target="_blank" rel="noopener">Ko-fi</a>.</p>
    </div>

    <div class="footer">Built with Flask, Azure OpenAI &amp; Openverse</div>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# AI: generate search queries using OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def ai_generate_queries(description, num_images):
    response = ai_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate image search queries for finding Creative Commons "
                    "images. Every query MUST directly describe the exact subject "
                    "the user requested. Do NOT broaden or generalize. "
                    "Only vary the angle, setting, or style — never the subject itself. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"I need images of exactly this: {description}\n"
                    f"Number of images needed: {num_images}\n\n"
                    "Generate a JSON object with:\n"
                    '- "queries": a list of search queries (2-5 words each). '
                    "EVERY query must include the core subject. "
                    "Only vary the perspective or context, not the subject.\n"
                    "- Make enough queries so results cover the total needed.\n\n"
                    "Example for 'catalytic converter':\n"
                    '{"queries": ["catalytic converter", "catalytic converter close up", '
                    '"catalytic converter removed", "catalytic converter metal"]}\n\n'
                    "BAD example (too broad, loses the subject):\n"
                    '{"queries": ["car exhaust", "engine parts", "vehicle underside"]}'
                ),
            },
        ],
    )

    text = response.choices[0].message.content.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Openverse: find CC-licensed images (free, no API key needed)
# ---------------------------------------------------------------------------
def search_openverse(query, page_size=20, page=1):
    url = "https://api.openverse.org/v1/images/"
    params = {
        "q": query,
        "license": "by,by-sa,by-nc,cc0",
        "page_size": min(page_size, 50),
        "page": page,
    }
    r = http_requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title", "") or "Untitled",
            "url": item.get("url", ""),
            "thumb": item.get("thumbnail", "") or item.get("url", ""),
            "width": item.get("width", 0),
            "height": item.get("height", 0),
        })
    return results


def collect_images(queries, total_needed):
    """Run multiple Openverse searches and collect unique image results."""
    all_images = []
    seen_urls = set()

    for i, query in enumerate(queries):
        if len(all_images) >= total_needed:
            break
        if i > 0:
            time.sleep(1.5)

        remaining = total_needed - len(all_images)
        page_size = min(remaining, 20)

        for page in range(1, 4):
            if len(all_images) >= total_needed:
                break
            try:
                results = search_openverse(query, page_size=page_size, page=page)
                if not results:
                    break
                for img in results:
                    if img["url"] not in seen_urls and len(all_images) < total_needed:
                        seen_urls.add(img["url"])
                        all_images.append(img)
                if page < 3:
                    time.sleep(1)
            except Exception:
                break

    return all_images


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    error = ""
    images = []
    queries_used = []
    description = ""
    num_images = 20
    images_json = ""

    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
        error = "Missing AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT in vars.env.txt"
    elif not AZURE_DEPLOYMENT_NAME:
        error = "Missing AZURE_DEPLOYMENT_NAME in vars.env.txt"

    if request.method == "POST" and not error:
        description = request.form.get("description", "").strip()
        num_images = int(request.form.get("num_images", 20))
        num_images = max(5, min(num_images, 50))

        if not description:
            error = "Please describe what images you need."
        else:
            try:
                ai_result = ai_generate_queries(description, num_images)
                queries_used = ai_result.get("queries", [description])

                images = collect_images(queries_used, num_images)

                if not images:
                    error = "No Creative Commons images found. Try a different description."
                else:
                    images_json = json.dumps([img["url"] for img in images])

            except json.JSONDecodeError:
                error = "AI returned an unexpected format. Please try again."
            except Exception as e:
                error = f"Error: {e}"

    visitor_count = _increment_counter() if request.method == "GET" else _read_counter()

    return render_template_string(
        PAGE,
        error=error,
        images=images,
        images_json=images_json,
        queries_used=queries_used,
        description=description,
        num_images=num_images,
        request_method=request.method,
        active_page="home",
        visitor_count=visitor_count,
    )


@app.route("/about")
def about():
    return render_template_string(ABOUT_PAGE)


@app.route("/download", methods=["POST"])
def download():
    """Download all found images and send them as a ZIP."""
    urls_json = request.form.get("urls", "[]")
    description = request.form.get("description", "dataset")

    try:
        urls = json.loads(urls_json)
    except json.JSONDecodeError:
        return "Invalid data", 400

    if not urls:
        return "No images to download", 400

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, url in enumerate(urls):
            try:
                r = http_requests.get(
                    url,
                    timeout=15,
                    headers={"User-Agent": "Mozilla/5.0 (DatasetBuilder/1.0)"},
                )
                r.raise_for_status()

                ct = r.headers.get("Content-Type", "")
                ext = ".jpg"
                if "png" in ct:
                    ext = ".png"
                elif "gif" in ct:
                    ext = ".gif"
                elif "webp" in ct:
                    ext = ".webp"

                zf.writestr(f"image_{i + 1:04d}{ext}", r.content)
            except Exception:
                continue

    zip_buffer.seek(0)

    safe_name = "".join(
        c if c.isalnum() or c in " -_" else "" for c in description
    )[:40].strip() or "dataset"

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_name}_images.zip",
    )


@app.route("/sitemap.xml")
def sitemap():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>http://localhost:5000/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>"""
    return xml, 200, {"Content-Type": "application/xml"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

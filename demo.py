import os
import io
import json
import time
import zipfile
import threading

import requests as http_requests
from flask import Flask, request, render_template_string, send_file, jsonify, Response
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

def _get_ai_client():
    key = os.environ.get("AZURE_OPENAI_KEY", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if key and endpoint:
        return AzureOpenAI(
            api_key=key,
            api_version="2024-10-21",
            azure_endpoint=endpoint,
        )
    return None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Persistent visitor counter (thread-safe, file-backed)
# ---------------------------------------------------------------------------
if os.environ.get("WEBSITE_SITE_NAME"):
    _counter_path = "/home/visitor_count.json"
else:
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
  <title>CuratorAI — AI-Powered Creative Commons Image Dataset Builder</title>
  <meta name="description" content="Build AI training datasets instantly. Describe the images you need and CuratorAI finds Creative Commons licensed pictures for you. Free to use.">
  <meta name="keywords" content="AI dataset builder, Creative Commons images, image dataset, machine learning, training data, free images, CuratorAI">
  <meta property="og:title" content="CuratorAI — AI Image Dataset Builder">
  <meta property="og:description" content="Describe what training images you need — AI finds Creative Commons pictures for you. Download as ZIP.">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://curatorai-pictures.azurewebsites.net">
  <meta name="google-site-verification" content="fmIlXnV59ndnxhx-qipF4K8tOs9gmfbUeFunue9c0OU" />
  <link rel="canonical" href="https://curatorai-pictures.azurewebsites.net/">
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

    .home-tabs { display: flex; gap: 8px; margin-bottom: 20px; }
    .home-tabs .btn { padding: 10px 22px; font-size: 0.9rem; }
    .home-audio-list { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; }
    .home-audio-item {
      display: flex; align-items: center; gap: 12px; padding: 10px 14px;
      background: rgba(255,255,255,0.05); border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);
    }
    .home-audio-item.deselected { opacity: 0.35; }
    .home-audio-item input[type="checkbox"] { width: 20px; height: 20px; accent-color: #7c3aed; flex-shrink: 0; }
    .home-audio-item .t { flex: 1; min-width: 0; font-size: 0.88rem; color: #e5e7eb; }
    .home-audio-item .meta { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
    .home-audio-item audio { height: 36px; max-width: 240px; flex-shrink: 0; }
  </style>
</head>
<body>
  <div class="loading-overlay" id="loadingOverlay">
    <div class="spinner"></div>
    <p id="loadingText">Searching...</p>
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
      <p id="homeTagline">Describe what you need &mdash; AI finds Creative Commons images or sounds for your dataset</p>
    </div>

    <div class="nav">
      <a href="/" class="{{ 'active' if active_page == 'home' }}">Home</a>
      <a href="/train" class="{{ 'active' if active_page == 'train' }}">Train</a>
      <a href="/about" class="{{ 'active' if active_page == 'about' }}">About</a>
    </div>

    {% if error %}
    <div class="error-box">{{ error }}</div>
    {% endif %}

    <!-- Search form -->
    <div class="card">
      <form method="post" action="/" id="homeSearchForm">
        <input type="hidden" name="media_type" id="homeMediaType" value="{{ media_type|default('image') }}" />
        <div class="home-tabs">
          <button type="button" class="btn btn-primary" id="tabImg" onclick="setHomeMedia('image')">Images</button>
          <button type="button" class="btn btn-outline" id="tabAud" onclick="setHomeMedia('audio')">Sounds</button>
        </div>
        <div class="form-row">
          <label for="description" id="homeLabelDesc">Describe the images you need</label>
          <textarea id="description" name="description"
            placeholder="e.g. Different breeds of dogs in outdoor settings for training a dog breed classifier">{{ description }}</textarea>
        </div>
        <div class="form-row">
          <label for="num_images" id="homeLabelNum">Number of items (5 &ndash; 50)</label>
          <input type="number" id="num_images" name="num_images"
                 value="{{ num_images }}" min="5" max="50" />
        </div>
        <button type="submit" class="btn btn-primary" id="homeSearchBtn">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"
               viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <span id="homeSearchBtnText">Search Images</span>
        </button>
      </form>
    </div>

    <!-- Results -->
    {% if images %}
    <div class="card">
      <div class="results-header">
        <div>
          {% if media_type == 'audio' %}
          <h2 style="font-size:1.2rem;">Found {{ images|length }} sound{{ "s" if images|length != 1 }}</h2>
          <div class="results-count">Creative Commons licensed &middot; preview with play, uncheck what you don&rsquo;t want, then download ZIP</div>
          {% else %}
          <h2 style="font-size:1.2rem;">Found {{ images|length }} image{{ "s" if images|length != 1 }}</h2>
          <div class="results-count">Creative Commons licensed &middot; uncheck images you don't want, then download</div>
          {% endif %}
        </div>
        <form method="post" action="/download" id="downloadForm">
          <input type="hidden" name="urls" id="selectedUrls" value="{{ images_json }}" />
          <input type="hidden" name="description" value="{{ description }}" />
          <input type="hidden" name="kind" id="downloadKind" value="{{ media_type|default('image') }}" />
          <button type="submit" class="btn btn-success">
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"
                 viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                 <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            <span id="downloadBtnLabel">Download ZIP</span>
          </button>
        </form>
      </div>

      <div class="selection-bar">
        <div><span class="count" id="selectedCount">{{ images|length }}</span> of {{ images|length }} selected</div>
        <div>
          <button type="button" onclick="homeToggleAll(true)">Select all</button>
          <button type="button" onclick="homeToggleAll(false)">Deselect all</button>
        </div>
      </div>

      {% if media_type == 'audio' %}
      <div class="home-audio-list" id="homeAudioList">
        {% for item in images %}
        <div class="home-audio-item" data-url="{{ item.url }}">
          <input type="checkbox" checked onchange="updateHomeSelection()" />
          <div class="t">{{ item.title }}
            <div class="meta">{% if item.duration %}{{ item.duration }}s{% endif %}{% if item.source %} &middot; {{ item.source }}{% endif %}</div>
          </div>
          <audio controls preload="none" src="/api/proxy-audio?url={{ item.url|urlencode }}"></audio>
        </div>
        {% endfor %}
      </div>
      {% else %}
      <div class="image-grid">
        {% for img in images %}
        <div class="image-card" data-url="{{ img.url }}">
          <input type="checkbox" checked onchange="updateSelection()" />
          <img src="{{ img.thumb }}" alt="{{ img.title }}" title="{{ img.title }}" loading="lazy"
               onclick="openLightbox('{{ img.url }}', '{{ img.title|e }}')" />
        </div>
        {% endfor %}
      </div>
      {% endif %}

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
      <div class="empty-state" id="homeEmptyMsg">No Creative Commons items found. Try a different description.</div>
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

    function setHomeMedia(t) {
      var hid = document.getElementById('homeMediaType');
      if (!hid) return;
      hid.value = t;
      var ti = document.getElementById('tabImg'), ta = document.getElementById('tabAud');
      if (ti) ti.className = t === 'image' ? 'btn btn-primary' : 'btn btn-outline';
      if (ta) ta.className = t === 'audio' ? 'btn btn-primary' : 'btn btn-outline';
      var ld = document.getElementById('homeLabelDesc');
      if (ld) ld.textContent = t === 'audio' ? 'Describe the sounds you need' : 'Describe the images you need';
      var ph = document.getElementById('description');
      if (ph) ph.placeholder = t === 'audio'
        ? 'e.g. bird calls, footsteps, rain — for a sound classification dataset'
        : 'e.g. Different breeds of dogs in outdoor settings for training a dog breed classifier';
      var btn = document.getElementById('homeSearchBtnText');
      if (btn) btn.textContent = t === 'audio' ? 'Search Sounds' : 'Search Images';
      var tag = document.getElementById('homeTagline');
      if (tag) tag.textContent = t === 'audio'
        ? 'Describe sounds you need — AI finds Creative Commons audio for your dataset'
        : 'Describe what you need — AI finds Creative Commons images or sounds for your dataset';
      var dk = document.getElementById('downloadKind');
      if (dk) dk.value = t;
    }

    function updateHomeSelection() {
      var items = document.querySelectorAll('.home-audio-item');
      if (!items.length) return;
      var urls = [];
      items.forEach(function(row) {
        var cb = row.querySelector('input[type="checkbox"]');
        if (cb.checked) {
          urls.push(row.dataset.url);
          row.classList.remove('deselected');
        } else row.classList.add('deselected');
      });
      var countEl = document.getElementById('selectedCount');
      if (countEl) countEl.textContent = urls.length;
      var urlsInput = document.getElementById('selectedUrls');
      if (urlsInput) urlsInput.value = JSON.stringify(urls);
    }

    function homeToggleAll(checked) {
      if (document.querySelector('.home-audio-item')) {
        document.querySelectorAll('.home-audio-item input[type="checkbox"]').forEach(function(cb) { cb.checked = checked; });
        updateHomeSelection();
      } else {
        toggleAll(checked);
      }
    }

    document.addEventListener('DOMContentLoaded', function() {
      var mt = document.getElementById('homeMediaType');
      if (mt && mt.value === 'audio') setHomeMedia('audio');
    });

    document.querySelectorAll('form').forEach(function(form) {
      form.addEventListener('submit', function() {
        var overlay = document.getElementById('loadingOverlay');
        var text = document.getElementById('loadingText');
        if (form.action.includes('/download')) {
          var dk = document.getElementById('downloadKind');
          var isAud = dk && dk.value === 'audio';
          text.textContent = isAud ? 'Downloading and zipping sounds...' : 'Downloading and zipping images...';
          overlay.classList.add('active');
          setTimeout(function() {
            overlay.classList.remove('active');
          }, 3000);
        } else {
          var hm = document.getElementById('homeMediaType');
          text.textContent = hm && hm.value === 'audio' ? 'Searching for sounds...' : 'Searching for images...';
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
  <title>About — CuratorAI | AI Image Dataset Builder</title>
  <meta name="description" content="Learn about CuratorAI, the AI-powered tool that helps you build Creative Commons image datasets for training machine learning models.">
  <link rel="canonical" href="https://curatorai-pictures.azurewebsites.net/about">
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
    .visitor-counter {
      position: fixed; top: 20px; left: 20px; z-index: 100;
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px; padding: 8px 14px; backdrop-filter: blur(8px);
      font-size: 0.75rem; color: #9ca3af;
    }
    .visitor-counter span { color: #a78bfa; font-weight: 600; }
  </style>
</head>
<body>
  <div class="visitor-counter">Visitors: <span>{{ visitor_count }}</span></div>
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
      <a href="/train">Train</a>
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


TRAIN_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Train — CuratorAI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      color: #e0e0e0;
      min-height: 100vh;
      padding: 32px 24px;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 40px; }
    .header h1 {
      font-size: 2.2rem; font-weight: 700;
      background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    .header p { color: #9ca3af; font-size: 0.95rem; }
    .nav { display: flex; justify-content: center; gap: 24px; margin-bottom: 28px; }
    .nav a {
      color: #9ca3af; text-decoration: none; font-size: 0.9rem; font-weight: 500;
      padding: 6px 0; border-bottom: 2px solid transparent; transition: color 0.2s, border-color 0.2s;
    }
    .nav a:hover, .nav a.active { color: #c4b5fd; border-bottom-color: #a78bfa; }
    .card {
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 32px; backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 24px;
    }
    .card h2 { font-size: 1.2rem; margin-bottom: 16px; color: #c4b5fd; }
    label { display: block; font-weight: 600; margin-bottom: 8px; font-size: 0.9rem; color: #c4b5fd; }
    input[type="text"], input[type="number"] {
      width: 100%; padding: 12px 16px; border: 1px solid rgba(255,255,255,0.15);
      border-radius: 12px; background: rgba(255,255,255,0.07); color: #f0f0f0;
      font-size: 1rem; font-family: inherit; outline: none; transition: border-color 0.2s;
    }
    input:focus { border-color: #a78bfa; box-shadow: 0 0 0 3px rgba(167,139,250,0.25); }
    input[type="number"] { max-width: 160px; }
    .form-row { margin-bottom: 16px; }
    .form-grid { display: grid; grid-template-columns: 1fr 200px 160px; gap: 12px; align-items: end; }
    .btn {
      display: inline-flex; align-items: center; gap: 8px; padding: 12px 24px; border: none;
      border-radius: 12px; font-size: 0.9rem; font-weight: 600; font-family: inherit;
      cursor: pointer; transition: transform 0.15s, box-shadow 0.2s;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-primary { background: linear-gradient(135deg, #7c3aed, #6366f1); color: #fff; }
    .btn-success { background: linear-gradient(135deg, #059669, #10b981); color: #fff; }
    .btn-danger { background: linear-gradient(135deg, #dc2626, #ef4444); color: #fff; padding: 6px 12px; font-size: 0.8rem; }
    .btn-outline {
      background: transparent; border: 1px solid rgba(167,139,250,0.3); color: #a78bfa;
      padding: 8px 16px; font-size: 0.85rem;
    }
    .btn-outline:hover { background: rgba(167,139,250,0.1); }

    .steps {
      display: flex; justify-content: center; gap: 8px; margin-bottom: 28px; font-size: 0.85rem;
    }
    .step { padding: 6px 16px; border-radius: 20px; color: #6b7280; background: rgba(255,255,255,0.05); }
    .step.active { background: rgba(167,139,250,0.2); color: #c4b5fd; font-weight: 600; }

    .class-bucket { margin-bottom: 20px; }
    .class-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 10px; padding: 8px 12px; background: rgba(167,139,250,0.1);
      border-radius: 10px;
    }
    .class-header h3 { font-size: 0.95rem; color: #c4b5fd; }
    .class-header span { font-size: 0.8rem; color: #9ca3af; }
    .thumb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 8px; }
    .thumb-item { position: relative; }
    .thumb-item img {
      width: 100%; height: 100px; object-fit: cover; border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.1); cursor: pointer;
    }
    .thumb-item .del-btn {
      position: absolute; top: 4px; right: 4px; width: 22px; height: 22px;
      background: rgba(220,38,38,0.85); color: #fff; border: none; border-radius: 50%;
      font-size: 14px; line-height: 22px; text-align: center; cursor: pointer;
      opacity: 0; transition: opacity 0.2s;
    }
    .thumb-item:hover .del-btn { opacity: 1; }

    .search-results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin: 16px 0; }
    .sr-item { position: relative; cursor: pointer; }
    .sr-item img {
      width: 100%; height: 120px; object-fit: cover; border-radius: 8px;
      border: 2px solid transparent; transition: border-color 0.2s, opacity 0.2s;
    }
    .sr-item.selected img { border-color: #a78bfa; }
    .sr-item.deselected img { opacity: 0.3; border-color: transparent; }
    .sr-item input[type="checkbox"] {
      position: absolute; top: 6px; right: 6px; width: 20px; height: 20px;
      accent-color: #7c3aed; cursor: pointer; z-index: 2;
    }

    .mode-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .mode-card {
      padding: 24px; border-radius: 16px; border: 2px solid rgba(255,255,255,0.1);
      background: rgba(255,255,255,0.04); cursor: pointer; text-align: center;
      transition: border-color 0.2s, background 0.2s;
    }
    .mode-card:hover { border-color: rgba(167,139,250,0.4); background: rgba(167,139,250,0.08); }
    .mode-card h3 { color: #c4b5fd; margin-bottom: 8px; }
    .mode-card p { color: #9ca3af; font-size: 0.85rem; }

    .progress-section { margin-top: 20px; }
    .progress-bar-bg {
      width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%; background: linear-gradient(90deg, #7c3aed, #34d399);
      border-radius: 4px; transition: width 0.3s;
    }
    .metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 16px; }
    .metric-box {
      text-align: center; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 10px;
    }
    .metric-box .val { font-size: 1.4rem; font-weight: 700; color: #a78bfa; }
    .metric-box .lbl { font-size: 0.75rem; color: #6b7280; margin-top: 4px; }
    .log-area {
      margin-top: 16px; max-height: 120px; overflow-y: auto; font-size: 0.8rem;
      color: #9ca3af; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px;
      font-family: monospace;
    }

    .annot-container { display: flex; gap: 16px; flex-wrap: wrap; }
    .annot-canvas-wrap {
      flex: 1; min-width: 500px; position: relative;
      background: rgba(0,0,0,0.3); border-radius: 12px; overflow: hidden;
    }
    .annot-canvas-wrap canvas { display: block; cursor: crosshair; }
    .annot-sidebar { width: 280px; flex-shrink: 0; }
    .annot-toolbar {
      display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap;
    }
    .annot-toolbar .btn { padding: 8px 12px; font-size: 0.8rem; }
    .annot-toolbar .btn.active-tool { background: rgba(167,139,250,0.3); border-color: #a78bfa; }
    .box-list { max-height: 300px; overflow-y: auto; }
    .box-item {
      display: flex; align-items: center; gap: 8px; padding: 6px 10px; margin-bottom: 4px;
      background: rgba(255,255,255,0.05); border-radius: 8px; font-size: 0.8rem;
    }
    .box-item.selected { background: rgba(167,139,250,0.15); border: 1px solid rgba(167,139,250,0.3); }
    .box-item select {
      flex: 1; padding: 4px 8px; background: rgba(255,255,255,0.1); color: #e0e0e0;
      border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; font-size: 0.8rem;
    }
    .img-nav { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .img-nav .btn { padding: 6px 14px; }
    .img-nav span { color: #9ca3af; font-size: 0.85rem; }

    .support { position: fixed; top: 20px; right: 20px; z-index: 100; }
    .btn-kofi {
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 20px; border: none;
      border-radius: 12px; background: linear-gradient(135deg, #ff5e5b, #ff9966); color: #fff;
      font-size: 0.85rem; font-weight: 600; font-family: inherit; text-decoration: none;
      cursor: pointer; transition: transform 0.15s, box-shadow 0.2s;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .btn-kofi:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(255,94,91,0.45); }
    .footer { text-align: center; margin-top: 16px; font-size: 0.75rem; color: #4b5563; }
    .audio-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
      background: rgba(255,255,255,0.05); border-radius: 10px; margin-bottom: 6px; }
    .audio-item.deselected { opacity: 0.3; }
    .audio-item input[type="checkbox"] { width: 18px; height: 18px; accent-color: #7c3aed; flex-shrink: 0; }
    .audio-item .audio-info { flex: 1; min-width: 0; }
    .audio-item .audio-title { font-size: 0.85rem; color: #e0e0e0; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis; }
    .audio-item .audio-meta { font-size: 0.75rem; color: #6b7280; }
    .audio-item audio { height: 32px; flex-shrink: 0; }
    .audio-thumb { display: flex; align-items: center; gap: 8px; padding: 4px 8px;
      background: rgba(255,255,255,0.05); border-radius: 8px; font-size: 0.8rem; color: #9ca3af; }
    .audio-thumb audio { height: 28px; }
    .hidden { display: none !important; }
    .status-msg { color: #9ca3af; font-size: 0.85rem; margin: 12px 0; }
    .visitor-counter {
      position: fixed; top: 20px; left: 20px; z-index: 100;
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px; padding: 8px 14px; backdrop-filter: blur(8px);
      font-size: 0.75rem; color: #9ca3af;
    }
    .visitor-counter span { color: #a78bfa; font-weight: 600; }
  </style>
</head>
<body>
  <div class="visitor-counter">Visitors: <span>{{ visitor_count }}</span></div>
  <div class="support">
    <a href="https://ko-fi.com/coolcoderme" target="_blank" rel="noopener" class="btn-kofi">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M23.881 8.948c-.773-4.085-4.859-4.593-4.859-4.593H.723c-.604 0-.679.798-.679.798s-.082 7.324-.022 11.822c.164 2.424 2.586 2.672 2.586 2.672s8.267-.023 11.966-.049c2.438-.426 2.683-2.566 2.658-3.734 4.352.24 7.422-2.831 6.649-6.916zm-11.062 3.511c-1.246 1.453-4.011 3.976-4.011 3.976s-.121.119-.31.023c-.076-.057-.108-.09-.108-.09-.443-.441-3.368-3.049-4.034-3.954-.709-.965-1.041-2.7-.091-3.71.951-1.01 3.005-1.086 4.363.407 0 0 1.565-1.782 3.468-.963 1.904.82 1.832 3.011.723 4.311zm6.173.478c-.928.116-1.682.028-1.682.028V7.284h1.77s1.971.551 1.971 2.638c0 1.913-.985 2.667-2.059 3.015z"/></svg>
      Support
    </a>
  </div>

  <div class="container">
    <div class="header">
      <h1>CuratorAI</h1>
      <p>Build datasets and train models — right in your browser</p>
    </div>
    <div class="nav">
      <a href="/">Home</a>
      <a href="/train" class="active">Train</a>
      <a href="/about">About</a>
    </div>

    <div class="steps">
      <div class="step active" id="step1">1. Build Dataset</div>
      <div class="step" id="step2">2. Train / Annotate</div>
      <div class="step" id="step3">3. Export</div>
    </div>

    <!-- ===== STEP 1: BUILD DATASET ===== -->
    <div id="buildView">
      <div class="card">
        <h2>Add a Class</h2>
        <div style="display:flex;gap:8px;margin-bottom:16px">
          <button class="btn btn-primary" id="typeImgBtn" onclick="setMediaType('image')" style="padding:8px 20px">Images</button>
          <button class="btn btn-outline" id="typeAudBtn" onclick="setMediaType('audio')" style="padding:8px 20px">Audio</button>
        </div>
        <div class="form-grid">
          <div class="form-row">
            <label for="className">Class name</label>
            <input type="text" id="className" placeholder="e.g. cats" />
          </div>
          <div class="form-row">
            <label for="classDesc">Description</label>
            <input type="text" id="classDesc" placeholder="e.g. cat meowing" />
          </div>
          <div class="form-row">
            <label for="classNum" id="classNumLabel">Items</label>
            <input type="number" id="classNum" value="20" min="5" max="50" />
          </div>
        </div>
        <button class="btn btn-primary" onclick="searchForClass()" id="searchBtn">Search</button>
        <div id="searchStatus" class="status-msg"></div>
      </div>

      <div id="searchResultsCard" class="card hidden">
        <h2>Search Results — <span id="srClassName"></span></h2>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <span class="status-msg"><span id="srSelectedCount">0</span> selected</span>
          <div style="display:flex;gap:6px">
            <button class="btn btn-outline" onclick="srToggleAll(true)">Select all</button>
            <button class="btn btn-outline" onclick="srToggleAll(false)">Deselect all</button>
          </div>
        </div>
        <div id="searchResultsGrid" class="search-results-grid"></div>
        <button class="btn btn-success" onclick="addToDataset()" style="margin-top:12px">Add Selected to Dataset</button>
      </div>

      <div id="datasetOverview" class="card hidden">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:10px">
          <h2 style="margin-bottom:0">Dataset Overview</h2>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <button class="btn btn-success" onclick="downloadDatasetZip()" id="downloadDatasetBtn" type="button">Download dataset ZIP</button>
            <button class="btn btn-primary" onclick="goToStep2()" id="continueBtn" disabled>Continue to Training</button>
          </div>
        </div>
        <p class="status-msg" style="margin-bottom:12px">Download saves all images or audio files in your dataset (same idea as Home &mdash; train locally without re-searching).</p>
        <div id="datasetBuckets"></div>
      </div>
    </div>

    <!-- ===== STEP 2: MODE SELECT ===== -->
    <div id="modeView" class="hidden">
      <div class="card">
        <h2>Choose Training Mode</h2>
        <div class="mode-cards">
          <div class="mode-card" onclick="startClassification()">
            <h3>Image Classification</h3>
            <p>Train a model to recognize which class an image belongs to. Training runs in your browser using your GPU.</p>
          </div>
          <div class="mode-card" onclick="startAnnotation()">
            <h3>Object Detection</h3>
            <p>Draw bounding boxes around objects. Export as a YOLO-format dataset for training locally.</p>
          </div>
        </div>
        <button class="btn btn-outline" onclick="goToStep1()" style="margin-top:16px">Back to Dataset</button>
      </div>
    </div>

    <!-- ===== CLASSIFICATION VIEW ===== -->
    <div id="classifyView" class="hidden">
      <div class="card">
        <h2>Classification Training</h2>
        <div style="display:flex;gap:16px;align-items:end;margin-bottom:16px;flex-wrap:wrap">
          <div class="form-row" style="margin-bottom:0">
            <label for="epochs">Epochs</label>
            <input type="number" id="epochs" value="20" min="5" max="100" />
          </div>
          <div class="form-row" style="margin-bottom:0">
            <label for="lr">Learning Rate</label>
            <input type="text" id="lr" value="0.001" />
          </div>
          <button class="btn btn-primary" onclick="trainModel()" id="trainBtn">Train Model</button>
          <button class="btn btn-outline" onclick="goToStep2()">Back</button>
        </div>
        <div id="trainingProgress" class="progress-section hidden">
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="progBar" style="width:0%"></div></div>
          <div class="metrics">
            <div class="metric-box"><div class="val" id="mEpoch">0</div><div class="lbl">Epoch</div></div>
            <div class="metric-box"><div class="val" id="mLoss">—</div><div class="lbl">Loss</div></div>
            <div class="metric-box"><div class="val" id="mAcc">—</div><div class="lbl">Accuracy</div></div>
          </div>
          <div class="log-area" id="trainLog"></div>
        </div>
        <div id="downloadSection" class="hidden" style="margin-top:20px">
          <button class="btn btn-success" onclick="downloadClassificationModel()">Download Model (TF.js)</button>
          <span class="status-msg" style="margin-left:12px">Includes a Python script to convert to TFLite</span>
        </div>
        <div id="quickTestSection" class="hidden" style="margin-top:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.1)">
          <h3 style="color:#c4b5fd;margin-bottom:10px;font-size:1.05rem">Quick test</h3>
          <p class="status-msg" style="margin-bottom:12px" id="quickTestHint">Upload a file to see predicted class and confidence.</p>
          <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
            <input type="file" id="quickTestFile" accept="image/*,audio/*" style="max-width:280px;color:#9ca3af" />
            <button class="btn btn-primary" onclick="runQuickTest()" id="quickTestBtn" type="button">Predict</button>
          </div>
          <div id="quickTestResult" class="status-msg" style="margin-top:14px;font-size:0.95rem;color:#c4b5fd"></div>
        </div>
      </div>
    </div>

    <!-- ===== ANNOTATION VIEW ===== -->
    <div id="annotateView" class="hidden">
      <div class="card">
        <h2>Bounding Box Annotation</h2>
        <div class="img-nav">
          <button class="btn btn-outline" onclick="annotPrev()">Prev</button>
          <span id="annotImgInfo">0 / 0</span>
          <button class="btn btn-outline" onclick="annotNext()">Next</button>
          <span style="flex:1"></span>
          <button class="btn btn-success" onclick="exportYOLO()">Export YOLO Dataset</button>
          <button class="btn btn-outline" onclick="goToStep2()">Back</button>
        </div>
        <div class="annot-container">
          <div class="annot-canvas-wrap">
            <canvas id="annotCanvas" width="700" height="500"></canvas>
          </div>
          <div class="annot-sidebar">
            <div class="annot-toolbar">
              <button class="btn btn-outline active-tool" id="toolDraw" onclick="setAnnotTool('draw')">Draw</button>
              <button class="btn btn-outline" id="toolSelect" onclick="setAnnotTool('select')">Select</button>
              <button class="btn btn-outline" onclick="annotDelete()">Delete</button>
              <button class="btn btn-outline" onclick="annotUndo()">Undo</button>
              <button class="btn btn-outline" onclick="annotRedo()">Redo</button>
            </div>
            <label>Boxes on this image:</label>
            <div class="box-list" id="boxList"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="footer">Built with Flask, Azure OpenAI, Openverse &amp; TensorFlow.js</div>
  </div>

<script>
// =====================================================================
// STATE
// =====================================================================
const dataset = {};
let searchResults = [];
let currentStep = 1;
let mediaType = 'image';
let trainingWasAudio = false;

function setMediaType(type) {
  mediaType = type;
  document.getElementById('typeImgBtn').className = type==='image' ? 'btn btn-primary' : 'btn btn-outline';
  document.getElementById('typeAudBtn').className = type==='audio' ? 'btn btn-primary' : 'btn btn-outline';
  document.getElementById('classNumLabel').textContent = 'Items';
  document.getElementById('classDesc').placeholder = type==='audio' ? 'e.g. cat meowing sounds' : 'e.g. house cats indoors';
}

// Annotation state
const annotations = {};
let allImages = [];
let annotIdx = 0;
let annotTool = 'draw';
let annotImg = null;
let annotBoxes = [];
let selectedBoxIdx = -1;
let isDrawing = false;
let drawStart = null;
let isDragging = false;
let dragOffset = {x:0,y:0};
let isResizing = false;
let resizeHandle = '';
let undoStack = [];
let redoStack = [];

// Classification state
let trainedModel = null;
let mobilenetModel = null;
const classNames = [];

// =====================================================================
// DATASET BUILDER
// =====================================================================
async function searchForClass() {
  const name = document.getElementById('className').value.trim();
  const desc = document.getElementById('classDesc').value.trim() || name;
  const num = parseInt(document.getElementById('classNum').value) || 20;
  if (!name) { alert('Enter a class name'); return; }

  const btn = document.getElementById('searchBtn');
  btn.disabled = true;
  document.getElementById('searchStatus').textContent = 'Searching...';
  try {
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: desc, num_images: Math.min(Math.max(num, 5), 50), type: mediaType})
    });
    const data = await resp.json();
    if (data.error) { document.getElementById('searchStatus').textContent = data.error; return; }
    searchResults = data.images.map(img => ({...img, selected: true}));
    document.getElementById('srClassName').textContent = name;
    renderSearchResults();
    document.getElementById('searchResultsCard').classList.remove('hidden');
    document.getElementById('searchStatus').textContent = '';
  } catch(e) {
    document.getElementById('searchStatus').textContent = 'Search failed: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderSearchResults() {
  const grid = document.getElementById('searchResultsGrid');
  grid.innerHTML = '';
  let selCount = 0;
  if (mediaType === 'audio') {
    grid.className = '';
    grid.style.display = 'flex';
    grid.style.flexDirection = 'column';
    searchResults.forEach((item, i) => {
      const div = document.createElement('div');
      div.className = 'audio-item' + (item.selected ? '' : ' deselected');
      const safeTitle = (item.title||'').replace(/"/g,'&quot;');
      div.innerHTML = '<input type="checkbox" ' + (item.selected?'checked':'') +
        ' onchange="srToggle('+i+')" />' +
        '<div class="audio-info"><div class="audio-title">'+safeTitle+'</div>' +
        '<div class="audio-meta">'+(item.duration?item.duration+'s':'')+(item.source?' · '+item.source:'')+'</div></div>' +
        '<audio controls preload="none" src="/api/proxy-audio?url='+encodeURIComponent(item.url)+'"></audio>';
      grid.appendChild(div);
      if (item.selected) selCount++;
    });
  } else {
    grid.className = 'search-results-grid';
    grid.style.display = '';
    grid.style.flexDirection = '';
    searchResults.forEach((img, i) => {
      const div = document.createElement('div');
      div.className = 'sr-item' + (img.selected ? ' selected' : ' deselected');
      div.innerHTML = '<input type="checkbox" ' + (img.selected ? 'checked' : '') +
        ' onchange="srToggle(' + i + ')" /><img src="' + img.thumb + '" alt="' + (img.title||'').replace(/"/g,'') +
        '" title="' + (img.title||'').replace(/"/g,'') + '" loading="lazy" />';
      div.querySelector('img').onclick = function() { srToggle(i); };
      grid.appendChild(div);
      if (img.selected) selCount++;
    });
  }
  document.getElementById('srSelectedCount').textContent = selCount;
}

function srToggle(i) {
  searchResults[i].selected = !searchResults[i].selected;
  renderSearchResults();
}
function srToggleAll(val) {
  searchResults.forEach(img => img.selected = val);
  renderSearchResults();
}

function addToDataset() {
  const name = document.getElementById('className').value.trim();
  if (!name) return;
  const selected = searchResults.filter(img => img.selected);
  if (!selected.length) { alert('Select at least one image'); return; }
  if (!dataset[name]) dataset[name] = [];
  selected.forEach(img => {
    if (!dataset[name].find(d => d.url === img.url)) dataset[name].push(img);
  });
  searchResults = [];
  document.getElementById('searchResultsCard').classList.add('hidden');
  document.getElementById('className').value = '';
  document.getElementById('classDesc').value = '';
  renderDataset();
}

function renderDataset() {
  const wrap = document.getElementById('datasetBuckets');
  const keys = Object.keys(dataset);
  if (!keys.length) {
    wrap.innerHTML = '<p class="status-msg">No classes added yet.</p>';
    document.getElementById('datasetOverview').classList.add('hidden');
    document.getElementById('continueBtn').disabled = true;
    return;
  }
  document.getElementById('datasetOverview').classList.remove('hidden');
  document.getElementById('continueBtn').disabled = keys.length < 2;
  wrap.innerHTML = '';
  keys.forEach(cls => {
    const bucket = document.createElement('div');
    bucket.className = 'class-bucket';
    const hdr = document.createElement('div');
    hdr.className = 'class-header';
    hdr.innerHTML = '<h3>' + cls + '</h3><span>' + dataset[cls].length + ' images</span>' +
      '<button class="btn btn-danger" onclick="removeClass(\\''+cls.replace(/'/g,"\\\\'")+
      '\\')">Remove class</button>';
    bucket.appendChild(hdr);
    const grid = document.createElement('div');
    grid.className = dataset[cls][0] && dataset[cls][0].source ? '' : 'thumb-grid';
    dataset[cls].forEach((img, i) => {
      const item = document.createElement('div');
      const isAudio = img.source === 'openverse' || img.source === 'freesound';
      if (isAudio) {
        item.className = 'audio-thumb';
        item.innerHTML = '<span>'+(img.title||'Audio').substring(0,30)+'</span>' +
          '<audio controls preload="none" src="/api/proxy-audio?url='+encodeURIComponent(img.url)+'"></audio>' +
          '<button class="btn btn-danger" onclick="removeImage(\\''+cls.replace(/'/g,"\\\\'")+
          '\\','+i+')">&times;</button>';
      } else {
        item.className = 'thumb-item';
        item.innerHTML = '<img src="'+img.thumb+'" alt="'+cls+'" />' +
          '<button class="del-btn" onclick="removeImage(\\''+cls.replace(/'/g,"\\\\'")+
          '\\','+i+')">&times;</button>';
      }
      grid.appendChild(item);
    });
    bucket.appendChild(grid);
    wrap.appendChild(bucket);
  });
}

function removeImage(cls, idx) {
  dataset[cls].splice(idx, 1);
  if (!dataset[cls].length) delete dataset[cls];
  renderDataset();
}
function removeClass(cls) {
  delete dataset[cls];
  renderDataset();
}

// =====================================================================
// STEP NAVIGATION
// =====================================================================
function setStep(n) {
  currentStep = n;
  document.querySelectorAll('.step').forEach((el,i) => el.classList.toggle('active', i===n-1));
  document.getElementById('buildView').classList.toggle('hidden', n!==1);
  document.getElementById('modeView').classList.toggle('hidden', n!==2);
  document.getElementById('classifyView').classList.toggle('hidden', n!==3 || !document.getElementById('classifyView').dataset.show);
  document.getElementById('annotateView').classList.toggle('hidden', n!==3 || !document.getElementById('annotateView').dataset.show);
}
function goToStep1() {
  document.getElementById('classifyView').dataset.show = '';
  document.getElementById('annotateView').dataset.show = '';
  setStep(1);
}
function goToStep2() {
  document.getElementById('classifyView').dataset.show = '';
  document.getElementById('annotateView').dataset.show = '';
  setStep(2);
}

// =====================================================================
// CLASSIFICATION TRAINING
// =====================================================================
function startClassification() {
  document.getElementById('classifyView').dataset.show = '1';
  document.getElementById('annotateView').dataset.show = '';
  setStep(3);
}

async function trainModel() {
  const keys = Object.keys(dataset);
  if (keys.length < 2) { alert('Need at least 2 classes'); return; }

  const btn = document.getElementById('trainBtn');
  btn.disabled = true;
  document.getElementById('trainingProgress').classList.remove('hidden');
  document.getElementById('downloadSection').classList.add('hidden');
  document.getElementById('quickTestSection').classList.add('hidden');
  trainingWasAudio = false;
  const fk = keys[0];
  if (fk && dataset[fk][0]) {
    const it = dataset[fk][0];
    trainingWasAudio = (it.source === 'openverse' || it.source === 'freesound');
  }
  const log = document.getElementById('trainLog');
  log.innerHTML = '';
  function addLog(msg) { log.innerHTML += msg + '\\n'; log.scrollTop = log.scrollHeight; }

  const totalEpochs = parseInt(document.getElementById('epochs').value) || 20;
  const learningRate = parseFloat(document.getElementById('lr').value) || 0.001;

  try {
    addLog('Loading MobileNet...');
    if (!mobilenetModel) {
      mobilenetModel = await mobilenet.load({version: 2, alpha: 1.0});
    }
    addLog('MobileNet loaded. Extracting features...');

    classNames.length = 0;
    keys.forEach(k => classNames.push(k));
    const numClasses = classNames.length;

    const features = [];
    const labels = [];
    let totalImages = 0;
    keys.forEach(k => totalImages += dataset[k].length);
    let processed = 0;

    for (let ci = 0; ci < keys.length; ci++) {
      const cls = keys[ci];
      for (const item of dataset[cls]) {
        try {
          const isAudio = item.source === 'openverse' || item.source === 'freesound';
          let imgEl;
          if (isAudio) {
            addLog('  Converting audio to spectrogram...');
            imgEl = await audioToSpectrogram(item.url);
          } else {
            imgEl = await loadImageEl(item.url);
          }
          const feat = mobilenetModel.infer(imgEl, true);
          features.push(feat);
          labels.push(ci);
          processed++;
          addLog('  Processed ' + processed + '/' + totalImages + ': ' + cls);
        } catch(e) {
          addLog('  Skipped item: ' + e.message);
        }
      }
    }

    if (features.length < 2) { addLog('ERROR: Not enough images loaded.'); btn.disabled = false; return; }

    const xs = tf.concat(features);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
    features.forEach(f => f.dispose());

    addLog('Building classifier (' + xs.shape[1] + ' features -> ' + numClasses + ' classes)...');
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [xs.shape[1]], units: 128, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.3}));
    model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));
    model.compile({optimizer: tf.train.adam(learningRate), loss: 'categoricalCrossentropy', metrics: ['accuracy']});

    addLog('Training for ' + totalEpochs + ' epochs...');
    await model.fit(xs, ys, {
      epochs: totalEpochs,
      batchSize: 16,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const pct = ((epoch+1)/totalEpochs*100).toFixed(0);
          document.getElementById('progBar').style.width = pct + '%';
          document.getElementById('mEpoch').textContent = (epoch+1) + '/' + totalEpochs;
          document.getElementById('mLoss').textContent = logs.loss.toFixed(4);
          document.getElementById('mAcc').textContent = (logs.acc*100).toFixed(1) + '%';
          addLog('Epoch ' + (epoch+1) + ' — loss: ' + logs.loss.toFixed(4) + '  acc: ' + (logs.acc*100).toFixed(1) + '%');
        }
      }
    });

    xs.dispose(); ys.dispose();
    trainedModel = model;
    addLog('Training complete!');
    document.getElementById('downloadSection').classList.remove('hidden');
    document.getElementById('quickTestSection').classList.remove('hidden');
    document.getElementById('quickTestHint').textContent = trainingWasAudio
      ? 'Upload an audio file (e.g. mp3, wav, ogg) to see predicted class.'
      : 'Upload an image file (jpg, png, gif, webp) to see predicted class.';
    document.getElementById('quickTestResult').textContent = '';
  } catch(e) {
    addLog('ERROR: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

function loadImageEl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = '/api/proxy-image?url=' + encodeURIComponent(url);
  });
}

async function audioToSpectrogram(url) {
  const resp = await fetch('/api/proxy-audio?url=' + encodeURIComponent(url));
  const arrayBuf = await resp.arrayBuffer();
  return await audioBufferToSpectrogram(arrayBuf);
}

async function audioBufferToSpectrogram(arrayBuf) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 22050});
  const copy = arrayBuf.slice(0);
  const audioBuf = await audioCtx.decodeAudioData(copy);
  const samples = audioBuf.getChannelData(0);
  const maxSamples = 22050 * 4;
  const data = samples.length > maxSamples ? samples.slice(0, maxSamples) : samples;

  const fftSize = 512;
  const hopSize = 256;
  const numFrames = Math.floor((data.length - fftSize) / hopSize);
  const numBins = fftSize / 2;
  if (numFrames < 2) throw new Error('Audio too short');

  const specData = [];
  const hann = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) hann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / fftSize));

  for (let f = 0; f < numFrames; f++) {
    const frame = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) frame[i] = (data[f*hopSize+i] || 0) * hann[i];
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    re.set(frame);
    fft(re, im, fftSize);
    const mags = new Float32Array(numBins);
    for (let i = 0; i < numBins; i++) mags[i] = Math.log1p(Math.sqrt(re[i]*re[i]+im[i]*im[i]));
    specData.push(mags);
  }

  const specCanvas = document.createElement('canvas');
  specCanvas.width = 224;
  specCanvas.height = 224;
  const sctx = specCanvas.getContext('2d');
  const imgData = sctx.createImageData(224, 224);
  let maxVal = 0;
  specData.forEach(col => col.forEach(v => { if(v>maxVal) maxVal=v; }));
  if (maxVal === 0) maxVal = 1;
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const fi = Math.floor(x / 224 * numFrames);
      const bi = numBins - 1 - Math.floor(y / 224 * numBins);
      const val = Math.round((specData[fi]?.[bi]||0) / maxVal * 255);
      const idx = (y * 224 + x) * 4;
      imgData.data[idx] = val;
      imgData.data[idx+1] = val * 0.6;
      imgData.data[idx+2] = val * 0.9;
      imgData.data[idx+3] = 255;
    }
  }
  sctx.putImageData(imgData, 0, 0);
  audioCtx.close();
  return specCanvas;
}

async function downloadDatasetZip() {
  const keys = Object.keys(dataset);
  if (!keys.length) { alert('Add at least one class first.'); return; }
  const zip = new JSZip();
  let n = 0;
  for (const cls of keys) {
    const safeDir = cls.replace(/[^a-zA-Z0-9_-]/g, '_').substring(0, 40) || 'class';
    for (let i = 0; i < dataset[cls].length; i++) {
      const item = dataset[cls][i];
      const isAudio = item.source === 'openverse' || item.source === 'freesound';
      try {
        const proxy = isAudio
          ? '/api/proxy-audio?url=' + encodeURIComponent(item.url)
          : '/api/proxy-image?url=' + encodeURIComponent(item.url);
        const resp = await fetch(proxy);
        if (!resp.ok) continue;
        const blob = await resp.blob();
        let ext = '.bin';
        const u = (item.url || '').toLowerCase();
        if (isAudio) {
          ext = u.includes('.mp3') ? '.mp3' : u.includes('.ogg') ? '.ogg' : u.includes('.wav') ? '.wav' : '.mp3';
        } else {
          ext = u.includes('.png') ? '.png' : u.includes('.gif') ? '.gif' : u.includes('.webp') ? '.webp' : '.jpg';
        }
        zip.file(safeDir + '/' + safeDir + '_' + String(i+1).padStart(3,'0') + ext, blob);
        n++;
      } catch(e) { console.warn(e); }
    }
  }
  if (!n) { alert('Could not download any files. Check your network.'); return; }
  const out = await zip.generateAsync({type: 'blob'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(out);
  a.download = 'curatorai_dataset.zip';
  a.click();
}

async function runQuickTest() {
  const fileInput = document.getElementById('quickTestFile');
  const file = fileInput.files && fileInput.files[0];
  const out = document.getElementById('quickTestResult');
  out.textContent = '';
  if (!trainedModel || !mobilenetModel) { out.textContent = 'Train a model first.'; return; }
  if (!file) { out.textContent = 'Choose an image or audio file.'; return; }
  document.getElementById('quickTestBtn').disabled = true;
  try {
    let inputTensor;
    if (trainingWasAudio) {
      const buf = await file.arrayBuffer();
      const canvas = await audioBufferToSpectrogram(buf);
      inputTensor = mobilenetModel.infer(canvas, true);
    } else {
      const img = new Image();
      const objUrl = URL.createObjectURL(file);
      await new Promise((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('Could not load image — use jpg, png, gif, or webp'));
        img.src = objUrl;
      });
      inputTensor = mobilenetModel.infer(img, true);
      URL.revokeObjectURL(objUrl);
    }
    const pred = trainedModel.predict(inputTensor);
    const data = await pred.data();
    inputTensor.dispose(); pred.dispose();
    let best = 0, bestP = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > bestP) { bestP = data[i]; best = i; }
    }
    out.innerHTML = '<strong>' + classNames[best] + '</strong> &mdash; ' + (bestP * 100).toFixed(1) + '% confidence';
  } catch(e) {
    out.textContent = 'Error: ' + e.message;
  } finally {
    document.getElementById('quickTestBtn').disabled = false;
  }
}

function fft(re, im, n) {
  if (n <= 1) return;
  const halfN = n / 2;
  const reEven = new Float32Array(halfN), imEven = new Float32Array(halfN);
  const reOdd = new Float32Array(halfN), imOdd = new Float32Array(halfN);
  for (let i = 0; i < halfN; i++) {
    reEven[i] = re[2*i]; imEven[i] = im[2*i];
    reOdd[i] = re[2*i+1]; imOdd[i] = im[2*i+1];
  }
  fft(reEven, imEven, halfN);
  fft(reOdd, imOdd, halfN);
  for (let k = 0; k < halfN; k++) {
    const angle = -2 * Math.PI * k / n;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const tRe = cos * reOdd[k] - sin * imOdd[k];
    const tIm = sin * reOdd[k] + cos * imOdd[k];
    re[k] = reEven[k] + tRe; im[k] = imEven[k] + tIm;
    re[k+halfN] = reEven[k] - tRe; im[k+halfN] = imEven[k] - tIm;
  }
}

async function downloadClassificationModel() {
  if (!trainedModel) return;
  const saveResult = await trainedModel.save(tf.io.withSaveHandler(async (artifacts) => {
    const weightData = new Uint8Array(artifacts.weightData);
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      weightsManifest: [{paths: ['weights.bin'], weights: artifacts.weightSpecs}],
      format: 'layers-model', generatedBy: 'CuratorAI'
    };

    const convertScript = [
      '# convert_to_tflite.py',
      '# Run: pip install tensorflowjs tensorflow',
      '# Then: python convert_to_tflite.py',
      'import tensorflowjs as tfjs', 'import tensorflow as tf', 'import os, json',
      'model = tfjs.converters.load_keras_model(os.path.join(".", "model.json"))',
      'converter = tf.lite.TFLiteConverter.from_keras_model(model)',
      'tflite = converter.convert()',
      'with open("model.tflite", "wb") as f: f.write(tflite)',
      'print("Saved model.tflite")'
    ].join('\\n');

    const classInfo = JSON.stringify({classes: classNames, input: 'MobileNetV2 1280-dim feature vector'}, null, 2);

    const zip = new JSZip();
    zip.file('model.json', JSON.stringify(modelJson));
    zip.file('weights.bin', weightData);
    zip.file('classes.json', classInfo);
    zip.file('convert_to_tflite.py', convertScript);
    const blob = await zip.generateAsync({type: 'blob'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'curatorai_classifier.zip';
    a.click();
    return {modelArtifactsInfo: {dateSaved: new Date(), modelTopologyType: 'JSON'}};
  }));
}

// =====================================================================
// ANNOTATION (BOUNDING BOXES)
// =====================================================================
function startAnnotation() {
  document.getElementById('annotateView').dataset.show = '1';
  document.getElementById('classifyView').dataset.show = '';
  allImages = [];
  Object.keys(dataset).forEach(cls => {
    dataset[cls].forEach(img => {
      if (!allImages.find(a => a.url === img.url)) allImages.push(img);
    });
  });
  annotIdx = 0;
  setStep(3);
  loadAnnotImage();
}

const canvas = document.getElementById('annotCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;

function loadAnnotImage() {
  if (!allImages.length) return;
  document.getElementById('annotImgInfo').textContent = (annotIdx+1) + ' / ' + allImages.length;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
    annotImg = img;
    const maxW = 700;
    const scale = maxW / img.naturalWidth;
    canvas.width = maxW;
    canvas.height = Math.round(img.naturalHeight * scale);
    annotBoxes = annotations[allImages[annotIdx].url] || [];
    selectedBoxIdx = -1;
    drawCanvas();
    renderBoxList();
  };
  img.src = '/api/proxy-image?url=' + encodeURIComponent(allImages[annotIdx].url);
}

function drawCanvas() {
  if (!annotImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(annotImg, 0, 0, canvas.width, canvas.height);
  annotBoxes.forEach((box, i) => {
    const x = box.x * canvas.width, y = box.y * canvas.height;
    const w = box.w * canvas.width, h = box.h * canvas.height;
    ctx.strokeStyle = i === selectedBoxIdx ? '#a78bfa' : '#34d399';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = i === selectedBoxIdx ? 'rgba(167,139,250,0.15)' : 'rgba(52,211,153,0.1)';
    ctx.fillRect(x, y, w, h);
    if (box.label) {
      ctx.fillStyle = i === selectedBoxIdx ? '#a78bfa' : '#34d399';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillText(box.label, x + 4, y - 4 > 12 ? y - 4 : y + 14);
    }
    if (i === selectedBoxIdx) {
      const hs = 5;
      ctx.fillStyle = '#a78bfa';
      [[x,y],[x+w,y],[x,y+h],[x+w,y+h],[x+w/2,y],[x+w/2,y+h],[x,y+h/2],[x+w,y+h/2]].forEach(([hx,hy]) => {
        ctx.fillRect(hx-hs/2, hy-hs/2, hs, hs);
      });
    }
  });
}

function renderBoxList() {
  const list = document.getElementById('boxList');
  const classKeys = Object.keys(dataset);
  list.innerHTML = '';
  annotBoxes.forEach((box, i) => {
    const item = document.createElement('div');
    item.className = 'box-item' + (i === selectedBoxIdx ? ' selected' : '');
    let opts = classKeys.map(c => '<option value="'+c+'"'+(box.label===c?' selected':'')+'>'+c+'</option>').join('');
    item.innerHTML = '<span>#'+(i+1)+'</span><select onchange="setBoxLabel('+i+',this.value)">'+
      '<option value="">(label)</option>'+opts+'</select>' +
      '<button class="btn btn-danger" onclick="deleteBox('+i+')">&times;</button>';
    item.onclick = function(e) { if(e.target.tagName==='DIV'||e.target.tagName==='SPAN'){selectedBoxIdx=i;drawCanvas();renderBoxList();} };
    list.appendChild(item);
  });
}

function setBoxLabel(i, val) { annotBoxes[i].label = val; saveAnnot(); drawCanvas(); }
function deleteBox(i) {
  pushUndo();
  annotBoxes.splice(i, 1);
  selectedBoxIdx = -1;
  saveAnnot(); drawCanvas(); renderBoxList();
}
function saveAnnot() {
  if (allImages.length) annotations[allImages[annotIdx].url] = annotBoxes;
}

function getCanvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return {x: (e.clientX - r.left) / canvas.width, y: (e.clientY - r.top) / canvas.height};
}
function getHandleAt(pos) {
  if (selectedBoxIdx < 0) return null;
  const box = annotBoxes[selectedBoxIdx];
  const hs = 8 / canvas.width;
  const corners = [
    {name:'tl',x:box.x,y:box.y},{name:'tr',x:box.x+box.w,y:box.y},
    {name:'bl',x:box.x,y:box.y+box.h},{name:'br',x:box.x+box.w,y:box.y+box.h},
    {name:'t',x:box.x+box.w/2,y:box.y},{name:'b',x:box.x+box.w/2,y:box.y+box.h},
    {name:'l',x:box.x,y:box.y+box.h/2},{name:'r',x:box.x+box.w,y:box.y+box.h/2},
  ];
  for (const c of corners) {
    if (Math.abs(pos.x-c.x)<hs && Math.abs(pos.y-c.y)<hs) return c.name;
  }
  return null;
}
function getBoxAt(pos) {
  for (let i = annotBoxes.length-1; i >= 0; i--) {
    const b = annotBoxes[i];
    if (pos.x>=b.x && pos.x<=b.x+b.w && pos.y>=b.y && pos.y<=b.y+b.h) return i;
  }
  return -1;
}

if (canvas) {
  canvas.addEventListener('mousedown', function(e) {
    const pos = getCanvasPos(e);
    if (annotTool === 'draw') {
      pushUndo();
      isDrawing = true;
      drawStart = pos;
    } else {
      const handle = getHandleAt(pos);
      if (handle) {
        pushUndo();
        isResizing = true;
        resizeHandle = handle;
        return;
      }
      const bi = getBoxAt(pos);
      if (bi >= 0) {
        selectedBoxIdx = bi;
        pushUndo();
        isDragging = true;
        dragOffset = {x: pos.x - annotBoxes[bi].x, y: pos.y - annotBoxes[bi].y};
        drawCanvas(); renderBoxList();
      } else {
        selectedBoxIdx = -1;
        drawCanvas(); renderBoxList();
      }
    }
  });
  canvas.addEventListener('mousemove', function(e) {
    const pos = getCanvasPos(e);
    if (isDrawing && drawStart) {
      drawCanvas();
      const x = Math.min(drawStart.x, pos.x) * canvas.width;
      const y = Math.min(drawStart.y, pos.y) * canvas.height;
      const w = Math.abs(pos.x - drawStart.x) * canvas.width;
      const h = Math.abs(pos.y - drawStart.y) * canvas.height;
      ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 2; ctx.setLineDash([5,5]);
      ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    } else if (isDragging && selectedBoxIdx >= 0) {
      const b = annotBoxes[selectedBoxIdx];
      b.x = Math.max(0, Math.min(pos.x - dragOffset.x, 1 - b.w));
      b.y = Math.max(0, Math.min(pos.y - dragOffset.y, 1 - b.h));
      drawCanvas();
    } else if (isResizing && selectedBoxIdx >= 0) {
      const b = annotBoxes[selectedBoxIdx];
      const h = resizeHandle;
      if (h.includes('r')) b.w = Math.max(0.02, pos.x - b.x);
      if (h.includes('l')) { const r = b.x+b.w; b.x = Math.min(pos.x, r-0.02); b.w = r-b.x; }
      if (h.includes('b')) b.h = Math.max(0.02, pos.y - b.y);
      if (h.includes('t')) { const bt = b.y+b.h; b.y = Math.min(pos.y, bt-0.02); b.h = bt-b.y; }
      drawCanvas();
    } else if (annotTool === 'select') {
      const handle = getHandleAt(pos);
      if (handle) {
        const cursors = {tl:'nwse-resize',tr:'nesw-resize',bl:'nesw-resize',br:'nwse-resize',
          t:'ns-resize',b:'ns-resize',l:'ew-resize',r:'ew-resize'};
        canvas.style.cursor = cursors[handle];
      } else if (getBoxAt(pos) >= 0) { canvas.style.cursor = 'move'; }
      else { canvas.style.cursor = 'default'; }
    }
  });
  canvas.addEventListener('mouseup', function(e) {
    if (isDrawing && drawStart) {
      const pos = getCanvasPos(e);
      const x = Math.min(drawStart.x, pos.x), y = Math.min(drawStart.y, pos.y);
      const w = Math.abs(pos.x - drawStart.x), h = Math.abs(pos.y - drawStart.y);
      if (w > 0.01 && h > 0.01) {
        annotBoxes.push({x, y, w, h, label: ''});
        selectedBoxIdx = annotBoxes.length - 1;
        saveAnnot(); renderBoxList();
      }
    }
    if (isDragging || isResizing) { saveAnnot(); renderBoxList(); }
    isDrawing = false; isDragging = false; isResizing = false; drawStart = null;
    drawCanvas();
  });
}

function setAnnotTool(tool) {
  annotTool = tool;
  document.getElementById('toolDraw').classList.toggle('active-tool', tool==='draw');
  document.getElementById('toolSelect').classList.toggle('active-tool', tool==='select');
  canvas.style.cursor = tool === 'draw' ? 'crosshair' : 'default';
}
function annotPrev() { if (annotIdx > 0) { saveAnnot(); annotIdx--; loadAnnotImage(); } }
function annotNext() { if (annotIdx < allImages.length-1) { saveAnnot(); annotIdx++; loadAnnotImage(); } }
function annotDelete() {
  if (selectedBoxIdx >= 0) deleteBox(selectedBoxIdx);
}

function pushUndo() {
  undoStack.push(JSON.parse(JSON.stringify(annotBoxes)));
  redoStack = [];
  if (undoStack.length > 50) undoStack.shift();
}
function annotUndo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.parse(JSON.stringify(annotBoxes)));
  annotBoxes = undoStack.pop();
  annotations[allImages[annotIdx].url] = annotBoxes;
  selectedBoxIdx = -1; drawCanvas(); renderBoxList();
}
function annotRedo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.parse(JSON.stringify(annotBoxes)));
  annotBoxes = redoStack.pop();
  annotations[allImages[annotIdx].url] = annotBoxes;
  selectedBoxIdx = -1; drawCanvas(); renderBoxList();
}

document.addEventListener('keydown', function(e) {
  if (document.getElementById('annotateView').classList.contains('hidden')) return;
  if (e.key === 'Delete' || e.key === 'Backspace') { if (e.target.tagName !== 'INPUT') annotDelete(); }
  if (e.ctrlKey && e.key === 'z') { e.preventDefault(); annotUndo(); }
  if (e.ctrlKey && e.key === 'y') { e.preventDefault(); annotRedo(); }
});

// =====================================================================
// YOLO EXPORT
// =====================================================================
async function exportYOLO() {
  saveAnnot();
  const classKeys = Object.keys(dataset);
  if (!classKeys.length) { alert('No classes defined'); return; }

  const zip = new JSZip();
  zip.file('classes.txt', classKeys.join('\\n'));

  let imgIndex = 0;
  for (const imgData of allImages) {
    const boxes = annotations[imgData.url] || [];
    const fname = 'image_' + String(imgIndex).padStart(4,'0');

    try {
      const resp = await fetch('/api/proxy-image?url=' + encodeURIComponent(imgData.url));
      const blob = await resp.blob();
      const ext = blob.type.includes('png') ? '.png' : '.jpg';
      zip.file('images/' + fname + ext, blob);
    } catch(e) { imgIndex++; continue; }

    let labelLines = '';
    boxes.forEach(box => {
      const ci = classKeys.indexOf(box.label);
      if (ci < 0) return;
      const cx = (box.x + box.w/2).toFixed(6);
      const cy = (box.y + box.h/2).toFixed(6);
      const bw = box.w.toFixed(6);
      const bh = box.h.toFixed(6);
      labelLines += ci + ' ' + cx + ' ' + cy + ' ' + bw + ' ' + bh + '\\n';
    });
    zip.file('labels/' + fname + '.txt', labelLines);
    imgIndex++;
  }

  const blob = await zip.generateAsync({type: 'blob'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'curatorai_yolo_dataset.zip';
  a.click();
}
</script>
<script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# AI: generate search queries using OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def ai_generate_queries(description, num_images):
    client = _get_ai_client()
    if not client:
        raise RuntimeError("Azure OpenAI not configured")
    deploy = os.environ.get("AZURE_DEPLOYMENT_NAME", "")
    response = client.chat.completions.create(
        model=deploy,
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
OPENVERSE_CLIENT_ID = "YvezdqGogM4j6FB8R1F59hfbDm01TBZorPhKqcjB"
OPENVERSE_CLIENT_SECRET = "RCU9m4vryjJvWF0HeiCstCYUOvdHDV6l7k5OCyxMrRNCco5FdGxjl5qLncUUD6u0igFzAmcvH9ZFGWEDB9gXA1v89AaVjf9YMFifjhGqHwMqcT6ZJ2farXw47fFv6Qhy"
_ov_token = {"access_token": None, "expires_at": 0}
_ov_token_lock = threading.Lock()


def _get_openverse_token():
    """Fetch or refresh the Openverse OAuth token."""
    now = time.time()
    with _ov_token_lock:
        if _ov_token["access_token"] and now < _ov_token["expires_at"] - 60:
            return _ov_token["access_token"]
        r = http_requests.post(
            "https://api.openverse.org/v1/auth_tokens/token/",
            data={
                "client_id": OPENVERSE_CLIENT_ID,
                "client_secret": OPENVERSE_CLIENT_SECRET,
                "grant_type": "client_credentials",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        _ov_token["access_token"] = data["access_token"]
        _ov_token["expires_at"] = now + data.get("expires_in", 43200)
        return _ov_token["access_token"]


def search_openverse(query, page_size=20, page=1):
    token = _get_openverse_token()
    url = "https://api.openverse.org/v1/images/"
    params = {
        "q": query,
        "license": "by,by-sa,by-nc,cc0",
        "page_size": min(page_size, 50),
        "page": page,
    }
    headers = {
        "User-Agent": "CuratorAI/1.0 (https://curatorai-pictures.azurewebsites.net)",
        "Authorization": f"Bearer {token}",
    }
    r = http_requests.get(url, params=params, headers=headers, timeout=15)
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
            time.sleep(0.3)

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
                    time.sleep(0.3)
            except Exception as exc:
                print(f"[Openverse] query={query!r} page={page} error={exc}")
                break

    return all_images


# ---------------------------------------------------------------------------
# Audio search: Openverse + Freesound
# ---------------------------------------------------------------------------
def search_openverse_audio(query, page_size=20, page=1):
    token = _get_openverse_token()
    url = "https://api.openverse.org/v1/audio/"
    params = {
        "q": query,
        "license": "by,by-sa,by-nc,cc0",
        "page_size": min(page_size, 50),
        "page": page,
    }
    headers = {
        "User-Agent": "CuratorAI/1.0",
        "Authorization": f"Bearer {token}",
    }
    r = http_requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", []):
        preview = ""
        if isinstance(item.get("url"), str):
            preview = item["url"]
        results.append({
            "title": item.get("title", "") or "Untitled",
            "url": preview,
            "duration": item.get("duration", 0),
            "source": "openverse",
        })
    return [r for r in results if r["url"]]


def search_freesound(query, page_size=20):
    fs_key = os.environ.get("FREESOUND_API_KEY", "")
    if not fs_key:
        return []
    url = "https://freesound.org/apiv2/search/text/"
    params = {
        "query": query,
        "token": fs_key,
        "fields": "id,name,previews,duration,license",
        "page_size": min(page_size, 50),
    }
    r = http_requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", []):
        previews = item.get("previews", {})
        preview = previews.get("preview-lq-mp3", "") or previews.get("preview-hq-mp3", "")
        if preview:
            results.append({
                "title": item.get("name", "Untitled"),
                "url": preview,
                "duration": round(item.get("duration", 0), 1),
                "source": "freesound",
            })
    return results


def collect_audio(queries, total_needed):
    """Search both Openverse and Freesound for audio."""
    all_audio = []
    seen_urls = set()

    for i, query in enumerate(queries):
        if len(all_audio) >= total_needed:
            break
        if i > 0:
            time.sleep(0.3)
        for search_fn in [search_openverse_audio, search_freesound]:
            if len(all_audio) >= total_needed:
                break
            try:
                results = search_fn(query, page_size=min(total_needed, 20))
                for audio in results:
                    if audio["url"] not in seen_urls and len(all_audio) < total_needed:
                        seen_urls.add(audio["url"])
                        all_audio.append(audio)
            except Exception:
                continue

    return all_audio


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
    media_type = "image"

    if not os.environ.get("AZURE_OPENAI_KEY") or not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        error = "Missing AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT"
    elif not os.environ.get("AZURE_DEPLOYMENT_NAME"):
        error = "Missing AZURE_DEPLOYMENT_NAME"

    if request.method == "POST" and not error:
        description = request.form.get("description", "").strip()
        num_images = int(request.form.get("num_images", 20))
        num_images = max(5, min(num_images, 50))
        media_type = request.form.get("media_type", "image").strip().lower()
        if media_type not in ("image", "audio"):
            media_type = "image"

        if not description:
            error = "Please describe what you are looking for."
        else:
            try:
                ai_result = ai_generate_queries(description, num_images)
                queries_used = ai_result.get("queries", [description])

                if media_type == "audio":
                    images = collect_audio(queries_used, num_images)
                    if not images:
                        error = "No Creative Commons audio found. Try a different description or add FREESOUND_API_KEY for more results."
                    else:
                        images_json = json.dumps([a["url"] for a in images])
                else:
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
        media_type=media_type,
    )


@app.route("/about")
def about():
    visitor_count = _increment_counter()
    return render_template_string(ABOUT_PAGE, visitor_count=visitor_count)


@app.route("/train")
def train():
    visitor_count = _read_counter()
    return render_template_string(TRAIN_PAGE, visitor_count=visitor_count)


@app.route("/api/search", methods=["POST"])
def api_search():
    """JSON endpoint for the training page to search images or audio."""
    data = request.get_json(silent=True) or {}
    description = data.get("description", "").strip()
    num_items = int(data.get("num_images", 20))
    num_items = max(5, min(num_items, 50))
    media_type = data.get("type", "image")

    if not description:
        return jsonify({"error": "Description is required"}), 400

    if not os.environ.get("AZURE_OPENAI_KEY") or not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return jsonify({"error": "AI service not configured"}), 500

    try:
        ai_result = ai_generate_queries(description, num_items)
        queries = ai_result.get("queries", [description])
        if media_type == "audio":
            items = collect_audio(queries, num_items)
        else:
            items = collect_images(queries, num_items)
        return jsonify({"images": items, "queries": queries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy-audio")
def proxy_audio():
    """Proxy external audio to avoid CORS issues for Web Audio API."""
    url = request.args.get("url", "")
    if not url.startswith("https://"):
        return "Invalid URL", 400
    try:
        r = http_requests.get(url, timeout=30, headers={"User-Agent": "CuratorAI/1.0"})
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "audio/mpeg")
        resp = Response(r.content, content_type=ct)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception:
        return "Audio not found", 404


@app.route("/api/proxy-image")
def proxy_image():
    """Proxy external images to avoid CORS issues for TF.js training."""
    url = request.args.get("url", "")
    if not url.startswith("https://"):
        return "Invalid URL", 400
    try:
        r = http_requests.get(url, timeout=15, headers={"User-Agent": "CuratorAI/1.0"})
        r.raise_for_status()
        resp = Response(r.content, content_type=r.headers.get("Content-Type", "image/jpeg"))
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception:
        return "Image not found", 404


@app.route("/debug")
def debug_check():
    import traceback
    out = []
    out.append(f"AZURE_OPENAI_KEY set: {bool(os.environ.get('AZURE_OPENAI_KEY'))}")
    out.append(f"AZURE_OPENAI_ENDPOINT set: {bool(os.environ.get('AZURE_OPENAI_ENDPOINT'))}")
    out.append(f"AZURE_DEPLOYMENT_NAME: {os.environ.get('AZURE_DEPLOYMENT_NAME', '(missing)')}")

    out.append("\n--- Testing AI ---")
    try:
        result = ai_generate_queries("dogs", 5)
        queries = result.get("queries", [])
        out.append(f"AI queries: {queries}")
    except Exception as e:
        out.append(f"AI error: {e}\n{traceback.format_exc()}")
        return "<pre>" + "\n".join(out) + "</pre>"

    out.append("\n--- Testing Openverse ---")
    try:
        imgs = search_openverse("dogs", page_size=3)
        out.append(f"Openverse returned {len(imgs)} results")
        for img in imgs:
            out.append(f"  - {img['title']}: {img['url'][:80]}")
    except Exception as e:
        out.append(f"Openverse error: {e}\n{traceback.format_exc()}")

    out.append("\n--- Testing collect_images ---")
    try:
        all_imgs = collect_images(queries[:2], 5)
        out.append(f"collect_images returned {len(all_imgs)} results")
    except Exception as e:
        out.append(f"collect_images error: {e}\n{traceback.format_exc()}")

    return "<pre>" + "\n".join(out) + "</pre>"


@app.route("/download", methods=["POST"])
def download():
    """Download found images or audio as a ZIP."""
    urls_json = request.form.get("urls", "[]")
    description = request.form.get("description", "dataset")
    kind = request.form.get("kind", "image").strip().lower()

    try:
        urls = json.loads(urls_json)
    except json.JSONDecodeError:
        return "Invalid data", 400

    if not urls:
        return "No files to download", 400

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, url in enumerate(urls):
            try:
                r = http_requests.get(
                    url,
                    timeout=30 if kind == "audio" else 15,
                    headers={"User-Agent": "Mozilla/5.0 (DatasetBuilder/1.0)"},
                )
                r.raise_for_status()

                ct = (r.headers.get("Content-Type") or "").lower()
                ulow = url.lower()
                if kind == "audio":
                    ext = ".mp3"
                    if "ogg" in ct or ulow.endswith(".ogg"):
                        ext = ".ogg"
                    elif "wav" in ct or ulow.endswith(".wav"):
                        ext = ".wav"
                    elif "mpeg" in ct or "mp3" in ct or ulow.endswith(".mp3"):
                        ext = ".mp3"
                    zf.writestr(f"audio_{i + 1:04d}{ext}", r.content)
                else:
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
    suffix = "sounds" if kind == "audio" else "images"

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_name}_{suffix}.zip",
    )


@app.route("/sitemap.xml")
def sitemap():
    base = "https://curatorai-pictures.azurewebsites.net"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{base}/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{base}/about</loc>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>
  <url>
    <loc>{base}/train</loc>
    <changefreq>weekly</changefreq>
    <priority>0.9</priority>
  </url>
</urlset>"""
    return xml, 200, {"Content-Type": "application/xml"}


@app.route("/robots.txt")
def robots():
    txt = """User-agent: *
Allow: /
Sitemap: https://curatorai-pictures.azurewebsites.net/sitemap.xml"""
    return txt, 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

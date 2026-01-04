"""
Expanded Preference Survey for ACL 2026 Resubmission.

This script creates a web-based pairwise comparison survey for collecting
user preferences on emotion visualization styles.

Requirements:
- Flask for web interface
- 30+ participants (vs current N=10)
- Per-user learning curves (2, 4, 8, 12 comparisons)
- Demographic stratification (age, accessibility needs)

Usage:
    python preference_survey.py --port 5001

Then share the URL with participants.
"""

from flask import Flask, render_template_string, request, jsonify, session
import json
import os
import random
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Survey configuration
EMOTIONS = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
STYLES = [
    {'id': 0, 'name': 'Subtle', 'description': 'Minimal visual emphasis'},
    {'id': 1, 'name': 'Bold', 'description': 'Strong font weight and size'},
    {'id': 2, 'name': 'Colorful', 'description': 'Vibrant emotion-coded colors'},
    {'id': 3, 'name': 'Animated', 'description': 'Motion and transitions'},
    {'id': 4, 'name': 'Classic', 'description': 'Traditional serif styling'},
    {'id': 5, 'name': 'Modern', 'description': 'Clean sans-serif design'},
]

N_COMPARISONS = 24  # Per user
DATA_DIR = '/home/muhiddin/ser/experiments/user_study/data'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class UserProfile:
    user_id: str
    age_group: str  # young (18-35), middle (36-55), senior (56+)
    culture: str  # western, eastern, other
    low_vision: bool
    color_blind: bool
    dyslexia: bool
    consent_given: bool
    start_time: str


@dataclass
class Comparison:
    user_id: str
    comparison_id: int
    emotion: str
    style_a: int
    style_b: int
    winner: int
    response_time_ms: int
    timestamp: str


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Visualization Preference Study</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 { color: #333; }
        .progress {
            height: 8px;
            background: #ddd;
            border-radius: 4px;
            margin: 16px 0;
        }
        .progress-bar {
            height: 100%;
            background: #4CAF50;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .style-option {
            display: inline-block;
            width: 45%;
            margin: 2%;
            padding: 20px;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .style-option:hover {
            border-color: #4CAF50;
            background: #f0fff0;
        }
        .style-option.selected {
            border-color: #4CAF50;
            background: #e8f5e9;
        }
        .emotion-label {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 16px;
        }
        .sample-text {
            font-size: 18px;
            padding: 16px;
            margin: 8px 0;
            border-radius: 4px;
        }
        /* Style previews */
        .style-0 { font-weight: normal; color: #666; }
        .style-1 { font-weight: bold; font-size: 22px; }
        .style-2 { color: #e91e63; background: #fce4ec; }
        .style-3 { animation: pulse 1s infinite; }
        .style-4 { font-family: Georgia, serif; font-style: italic; }
        .style-5 { font-family: 'Helvetica Neue', sans-serif; letter-spacing: 1px; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 16px;
        }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .form-group {
            margin: 16px 0;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        select, input[type="checkbox"] {
            padding: 8px;
            font-size: 16px;
        }
        .checkbox-group label {
            display: inline;
            font-weight: normal;
            margin-left: 8px;
        }
    </style>
</head>
<body>
    <div id="app"></div>

    <script>
        const API_BASE = '';
        let state = {
            phase: 'consent',  // consent, demographics, comparison, complete
            userId: null,
            currentComparison: 0,
            totalComparisons: {{ n_comparisons }},
            comparisonStartTime: null,
            comparisons: []
        };

        function render() {
            const app = document.getElementById('app');

            switch(state.phase) {
                case 'consent':
                    app.innerHTML = renderConsent();
                    break;
                case 'demographics':
                    app.innerHTML = renderDemographics();
                    break;
                case 'comparison':
                    app.innerHTML = renderComparison();
                    break;
                case 'complete':
                    app.innerHTML = renderComplete();
                    break;
            }
        }

        function renderConsent() {
            return `
                <div class="card">
                    <h1>Emotion Visualization Preference Study</h1>
                    <p>Thank you for participating in this research study on emotion visualization preferences.</p>

                    <h3>What you'll do:</h3>
                    <ul>
                        <li>Answer a few demographic questions (2 minutes)</li>
                        <li>Compare pairs of visualization styles (8-10 minutes)</li>
                        <li>Choose which style you prefer for displaying emotions in subtitles</li>
                    </ul>

                    <h3>Privacy:</h3>
                    <ul>
                        <li>Your responses are anonymous</li>
                        <li>No personally identifying information is collected</li>
                        <li>Data will be used for academic research only</li>
                    </ul>

                    <p><strong>IRB Protocol #2024-0847</strong></p>

                    <div class="form-group checkbox-group">
                        <input type="checkbox" id="consent" onchange="checkConsent()">
                        <label for="consent">I consent to participate in this study</label>
                    </div>

                    <button id="startBtn" disabled onclick="startStudy()">Begin Study</button>
                </div>
            `;
        }

        function checkConsent() {
            document.getElementById('startBtn').disabled = !document.getElementById('consent').checked;
        }

        function startStudy() {
            state.phase = 'demographics';
            render();
        }

        function renderDemographics() {
            return `
                <div class="card">
                    <h2>About You</h2>
                    <p>Please answer these brief questions to help us understand preference patterns.</p>

                    <div class="form-group">
                        <label for="age">Age Group:</label>
                        <select id="age" required>
                            <option value="">Select...</option>
                            <option value="young">18-35 years</option>
                            <option value="middle">36-55 years</option>
                            <option value="senior">56+ years</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="culture">Cultural Background:</label>
                        <select id="culture" required>
                            <option value="">Select...</option>
                            <option value="western">Western (Americas, Europe, Australia)</option>
                            <option value="eastern">Eastern (Asia, Middle East)</option>
                            <option value="other">Other / Mixed</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Accessibility Needs (check all that apply):</label>
                        <div class="checkbox-group">
                            <input type="checkbox" id="low_vision">
                            <label for="low_vision">Low vision / visual impairment</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="color_blind">
                            <label for="color_blind">Color blindness</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="dyslexia">
                            <label for="dyslexia">Dyslexia</label>
                        </div>
                    </div>

                    <button onclick="submitDemographics()">Continue to Comparisons</button>
                </div>
            `;
        }

        async function submitDemographics() {
            const age = document.getElementById('age').value;
            const culture = document.getElementById('culture').value;

            if (!age || !culture) {
                alert('Please fill in all required fields');
                return;
            }

            const profile = {
                age_group: age,
                culture: culture,
                low_vision: document.getElementById('low_vision').checked,
                color_blind: document.getElementById('color_blind').checked,
                dyslexia: document.getElementById('dyslexia').checked
            };

            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(profile)
            });

            const data = await response.json();
            state.userId = data.user_id;
            state.comparisons = data.comparisons;
            state.phase = 'comparison';
            state.comparisonStartTime = Date.now();
            render();
        }

        function renderComparison() {
            const comp = state.comparisons[state.currentComparison];
            const progress = ((state.currentComparison) / state.totalComparisons) * 100;

            const emotions = {{ emotions | safe }};
            const styles = {{ styles | safe }};

            const emotionColors = {
                'happiness': '#FFD700',
                'sadness': '#4169E1',
                'anger': '#DC143C',
                'fear': '#8B008B',
                'surprise': '#FF8C00',
                'neutral': '#808080'
            };

            return `
                <div class="card">
                    <div class="progress">
                        <div class="progress-bar" style="width: ${progress}%"></div>
                    </div>
                    <p>Comparison ${state.currentComparison + 1} of ${state.totalComparisons}</p>

                    <div class="emotion-label">
                        For <span style="color: ${emotionColors[comp.emotion]}">${comp.emotion.toUpperCase()}</span> emotions, which style do you prefer?
                    </div>

                    <p>"I can't believe this is happening right now!"</p>

                    <div>
                        <div class="style-option" id="optionA" onclick="selectOption('A')">
                            <div class="sample-text style-${comp.style_a}">
                                "${styles[comp.style_a].name}"
                            </div>
                            <small>${styles[comp.style_a].description}</small>
                        </div>

                        <div class="style-option" id="optionB" onclick="selectOption('B')">
                            <div class="sample-text style-${comp.style_b}">
                                "${styles[comp.style_b].name}"
                            </div>
                            <small>${styles[comp.style_b].description}</small>
                        </div>
                    </div>

                    <button id="nextBtn" disabled onclick="submitComparison()">Next</button>
                </div>
            `;
        }

        let selectedOption = null;

        function selectOption(option) {
            selectedOption = option;
            document.querySelectorAll('.style-option').forEach(el => el.classList.remove('selected'));
            document.getElementById('option' + option).classList.add('selected');
            document.getElementById('nextBtn').disabled = false;
        }

        async function submitComparison() {
            const comp = state.comparisons[state.currentComparison];
            const responseTime = Date.now() - state.comparisonStartTime;

            const winner = selectedOption === 'A' ? comp.style_a : comp.style_b;

            await fetch('/api/comparison', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    user_id: state.userId,
                    comparison_id: state.currentComparison,
                    emotion: comp.emotion,
                    style_a: comp.style_a,
                    style_b: comp.style_b,
                    winner: winner,
                    response_time_ms: responseTime
                })
            });

            state.currentComparison++;
            selectedOption = null;

            if (state.currentComparison >= state.totalComparisons) {
                state.phase = 'complete';
            } else {
                state.comparisonStartTime = Date.now();
            }

            render();
        }

        function renderComplete() {
            return `
                <div class="card">
                    <h1>Thank You!</h1>
                    <p>Your responses have been recorded.</p>
                    <p>Your participation helps us understand how people prefer to see emotions visualized in subtitles.</p>
                    <p><strong>Participant ID: ${state.userId}</strong></p>
                    <p>If you have any questions, please contact the research team.</p>
                </div>
            `;
        }

        // Initialize
        render();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        n_comparisons=N_COMPARISONS,
        emotions=json.dumps(EMOTIONS),
        styles=json.dumps(STYLES)
    )


@app.route('/api/start', methods=['POST'])
def start_study():
    data = request.json

    user_id = str(uuid.uuid4())[:8]

    # Create user profile
    profile = UserProfile(
        user_id=user_id,
        age_group=data['age_group'],
        culture=data['culture'],
        low_vision=data.get('low_vision', False),
        color_blind=data.get('color_blind', False),
        dyslexia=data.get('dyslexia', False),
        consent_given=True,
        start_time=datetime.now().isoformat()
    )

    # Save profile
    profile_path = os.path.join(DATA_DIR, f'{user_id}_profile.json')
    with open(profile_path, 'w') as f:
        json.dump(asdict(profile), f, indent=2)

    # Generate comparisons (balanced across emotions and style pairs)
    comparisons = []
    for i in range(N_COMPARISONS):
        emotion = EMOTIONS[i % len(EMOTIONS)]
        style_a = random.randint(0, 5)
        style_b = random.randint(0, 5)
        while style_b == style_a:
            style_b = random.randint(0, 5)

        comparisons.append({
            'emotion': emotion,
            'style_a': style_a,
            'style_b': style_b
        })

    return jsonify({
        'user_id': user_id,
        'comparisons': comparisons
    })


@app.route('/api/comparison', methods=['POST'])
def save_comparison():
    data = request.json

    comparison = Comparison(
        user_id=data['user_id'],
        comparison_id=data['comparison_id'],
        emotion=data['emotion'],
        style_a=data['style_a'],
        style_b=data['style_b'],
        winner=data['winner'],
        response_time_ms=data['response_time_ms'],
        timestamp=datetime.now().isoformat()
    )

    # Append to user's comparison file
    comp_path = os.path.join(DATA_DIR, f"{data['user_id']}_comparisons.jsonl")
    with open(comp_path, 'a') as f:
        f.write(json.dumps(asdict(comparison)) + '\n')

    return jsonify({'status': 'ok'})


@app.route('/api/stats')
def get_stats():
    """Get current participation statistics."""
    profiles = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('_profile.json'):
            with open(os.path.join(DATA_DIR, f)) as fp:
                profiles.append(json.load(fp))

    # Count by demographics
    age_counts = {}
    culture_counts = {}
    accessibility_counts = {'low_vision': 0, 'color_blind': 0, 'dyslexia': 0}

    for p in profiles:
        age_counts[p['age_group']] = age_counts.get(p['age_group'], 0) + 1
        culture_counts[p['culture']] = culture_counts.get(p['culture'], 0) + 1
        if p.get('low_vision'):
            accessibility_counts['low_vision'] += 1
        if p.get('color_blind'):
            accessibility_counts['color_blind'] += 1
        if p.get('dyslexia'):
            accessibility_counts['dyslexia'] += 1

    return jsonify({
        'total_participants': len(profiles),
        'by_age': age_counts,
        'by_culture': culture_counts,
        'accessibility': accessibility_counts,
        'target': 30
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Preference Survey for ACL 2026 Resubmission")
    print(f"{'='*60}")
    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"Share this URL with participants to collect data")
    print(f"\nTarget: 30+ participants")
    print(f"Current: Check /api/stats for progress")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)

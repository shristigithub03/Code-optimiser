# List of required packages
REQUIRED_PACKAGES = [
    'flask',
    'numpy',
    'pandas',
    'faker',
    'requests',
    'pillow',
    'networkx',
    'matplotlib',
    'duckduckgo_search',
    'google.generativeai',
    'sklearn'
]

import importlib.util
import subprocess
import os
import sys

# Check and install missing packages
def install_packages():
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(package.split('.')[0])
            if spec is None:
                missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
            print("All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install some dependencies. Please install them manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)

install_packages()

import os
import json
import zipfile
import numpy as np
import pandas as pd
import random
import requests
from io import BytesIO
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from duckduckgo_search import DDGS
from typing import Iterator, Dict, Optional, Tuple
from flask import Flask, render_template_string, request, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_files'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DatasetGenerator:
    def __init__(self):
        self.faker = Faker()
        
        # API Keys (replace with your actual keys - but this is not recommended for production)
        
        
        # Initialize AI services if keys provided
        if self.google_api_key and self.google_api_key != "YOUR_GOOGLE_API_KEY":
            try:
                genai.configure(api_key=self.google_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.gemini_model = None
        
        if self.hf_api_key and self.hf_api_key != "YOUR_HUGGINGFACE_KEY":
            self.hf_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            self.hf_headers = {"Authorization": f"Bearer {self.hf_api_key}"}

    def generate_data(self, params):
        data_type = params['data_type']
        
        if data_type == 'tabular':
            return self._generate_tabular_data(params)
        elif data_type == 'time_series':
            return self._generate_time_series_data(params)
        elif data_type == 'text':
            return self._generate_text_data(params)
        elif data_type == 'image':
            return self._generate_image_data(params)
        elif data_type == 'graph':
            return self._generate_graph_data(params)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def _generate_tabular_data(self, params):
        fields = params['fields']
        size = params['size']
        
        if 'target' in fields or 'label' in fields:
            if random.choice([True, False]):
                X, y = make_classification(
                    n_samples=size,
                    n_features=len(fields)-1,
                    n_informative=max(2, (len(fields)-1)//2),
                    n_classes=random.randint(2, 5),
                    random_state=42
                )
                data = pd.DataFrame(X, columns=[f for f in fields if f not in ['target', 'label']])
                data['target'] = y
            else:
                X, y = make_regression(
                    n_samples=size,
                    n_features=len(fields)-1,
                    n_informative=max(2, (len(fields)-1)//2),
                    noise=10,
                    random_state=42
                )
                data = pd.DataFrame(X, columns=[f for f in fields if f not in ['target', 'label']])
                data['target'] = y
        else:
            data = {}
            for field in fields:
                generator = self._get_field_generator(field.lower())
                data[field] = [generator() for _ in range(size)]
            data = pd.DataFrame(data)
        
        return data

    def _generate_time_series_data(self, params):
        fields = params['fields']
        size = params['size']
        
        if 'timestamp' not in fields and 'date' not in fields and 'time' not in fields:
            fields.append('timestamp')
        
        start_date = self.faker.date_time_this_decade()
        timestamps = [start_date + timedelta(days=i) for i in range(size)]
        
        data = {'timestamp': [ts.isoformat() for ts in timestamps]}
        
        for field in fields:
            if field.lower() in ['timestamp', 'date', 'time']:
                continue
                
            pattern_type = random.choice(['random_walk', 'seasonal', 'trend', 'step'])
            values = self._generate_time_series_pattern(size, pattern_type)
            data[field] = self._scale_values(values, field.lower())
        
        return pd.DataFrame(data)

    def _generate_text_data(self, params):
        fields = params['fields']
        size = params['size']
        topic = params.get('topic', 'general')
        
        data = {}
        
        for field in fields:
            field_lower = field.lower()
            if 'id' in field_lower:
                data[field] = [self.faker.unique.uuid4() for _ in range(size)]
            elif 'text' in field_lower or 'content' in field_lower:
                data[field] = [self._generate_topic_text(topic) for _ in range(size)]
            elif 'sentiment' in field_lower:
                data[field] = [random.choice(['positive', 'negative', 'neutral']) for _ in range(size)]
            else:
                data[field] = [self.faker.word() for _ in range(size)]
        
        return pd.DataFrame(data)

    def _generate_topic_text(self, topic):
        if hasattr(self, 'gemini_model'):
            try:
                response = self.gemini_model.generate_content(
                    f"Generate 2-3 coherent, realistic sentences about {topic}. "
                    "Use natural everyday language. Output ONLY the raw text."
                )
                return response.text.strip('"\' \n')
            except Exception as e:
                print(f"Error generating text with Gemini: {e}")
        
        topic_keywords = [word for word in topic.split() if len(word) > 3][:2]
        structures = [
            f"This {topic_keywords[0]} {random.choice(['delivers','offers','provides'])} "
            f"{random.choice(['excellent','good','solid'])} {random.choice(['performance','results','quality'])}. "
            f"{self.faker.sentence(nb_words=8)}",
        
            f"For {random.choice(['beginners','professionals'])} in {topic}, "
            f"the {random.choice(['key feature','main advantage'])} is "
            f"{self.faker.sentence(nb_words=6).lower()}",
        
            f"Many users find this {topic_keywords[-1]} {random.choice(['helpful','frustrating'])} "
            f"because {self.faker.sentence(nb_words=8)}"
        ]
        return random.choice(structures)

    def _generate_image_data(self, params):
        images = []
        remaining = params['size']
        
        used_urls = set()
        
        if not params.get('prefer_ai', False):
            try:
                with DDGS() as ddgs:
                    image_results = list(ddgs.images(
                        params['topic'],
                        max_results=min(100, remaining*3),
                        region="wt-wt",
                        safesearch="off" if not params.get('safesearch', True) else "moderate"
                    ))
                    
                    random.shuffle(image_results)
                    
                    for result in image_results:
                        if remaining <= 0:
                            break
                        if result['image'] in used_urls:
                            continue
                            
                        img = self._download_image(result['image'])
                        if img:
                            images.append(img.resize(self._parse_resolution(params.get('resolution', '512x512'))))
                            used_urls.add(result['image'])
                            remaining -= 1
            except Exception as e:
                print(f"Web image search failed: {e}")

        if remaining > 0 and hasattr(self, 'hf_headers'):
            try:
                for i in range(remaining):
                    prompt = f"{params['topic']} - variation {i+1}"
                    img = self._generate_ai_image(prompt)
                    if img:
                        images.append(img.resize(self._parse_resolution(params.get('resolution', '512x512'))))
                        remaining -= 1
            except Exception as e:
                print(f"AI generation failed: {e}")

        if remaining > 0:
            for i in range(remaining):
                img = self._generate_placeholder_image(params['topic'], index=i)
                images.append(img)
        
        return self._package_images(images, params)

    def _generate_placeholder_image(self, topic, index=0):
        img = Image.new('RGB', (512, 512), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default().font_variant(size=20)
            text = f"Placeholder for: {topic}\nVariation #{index + 1}"
            
            text_width, text_height = draw.textsize(text, font=font)
            x = (512 - text_width) // 2
            y = (512 - text_height) // 2
            
            draw.text((x, y), text, fill=(100, 100, 100), font=font)
        except:
            pass
        
        return img

    def _generate_graph_data(self, params):
        size = params['size']
        G = nx.erdos_renyi_graph(n=size, p=0.1)
        
        for field in params['fields']:
            if field.lower() == 'id':
                for node in G.nodes():
                    G.nodes[node][field] = str(node)
            elif field.lower() == 'community':
                communities = {n: random.randint(1, 3) for n in G.nodes()}
                nx.set_node_attributes(G, communities, field)
            else:
                nx.set_node_attributes(G, {n: random.randint(1, 100) for n in G.nodes()}, field)
        
        return G

    def _generate_time_series_pattern(self, size, pattern_type):
        if pattern_type == 'random_walk':
            return np.cumsum(np.random.normal(0, 1, size))
        elif pattern_type == 'seasonal':
            return np.sin(np.linspace(0, 10*np.pi, size)) * 10 + np.random.normal(0, 1, size)
        elif pattern_type == 'trend':
            return np.linspace(0, 100, size) + np.random.normal(0, 5, size)
        else:
            step_points = sorted(random.sample(range(size), random.randint(2, 5)))
            step_values = np.random.uniform(10, 100, len(step_points))
            values = np.zeros(size)
            current_value = step_values[0]
            step_idx = 0
            for i in range(size):
                if step_idx < len(step_points) and i >= step_points[step_idx]:
                    current_value = step_values[step_idx]
                    step_idx += 1
                values[i] = current_value + np.random.normal(0, 1)
            return values

    def _scale_values(self, values, field_name):
        if 'temp' in field_name:
            return np.round((values - np.min(values)) / (np.max(values) - np.min(values)) * 30 + 10, 2)
        elif 'price' in field_name or 'amount' in field_name:
            return np.round((values - np.min(values)) / (np.max(values) - np.min(values)) * 900 + 100, 2)
        elif 'percentage' in field_name:
            return np.round((values - np.min(values)) / (np.max(values) - np.min(values)) * 100, 2)
        return np.round(values, 2)

    def _get_field_generator(self, field_name):
        if 'name' in field_name:
            return self.faker.name
        elif 'date' in field_name or 'time' in field_name:
            return lambda: self.faker.date_time_this_decade().isoformat()
        elif 'email' in field_name:
            return self.faker.email
        elif 'amount' in field_name or 'price' in field_name:
            return lambda: round(random.uniform(1, 1000), 2)
        elif 'bool' in field_name:
            return lambda: random.choice([True, False])
        else:
            return lambda: random.randint(1, 100)

    def _download_image(self, url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            if not response.headers.get('Content-Type', '').startswith('image/'):
                raise ValueError("URL does not point to an image")
                
            img = Image.open(BytesIO(response.content))
            img.verify()
            img = Image.open(BytesIO(response.content))
            
            return img
            
        except Exception as e:
            print(f"Failed to download {url[:50]}...: {e}")
            return None

    def _generate_ai_image(self, prompt):
        try:
            response = requests.post(
                self.hf_api_url,
                headers=self.hf_headers,
                json={"inputs": prompt},
                timeout=30
            )
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"AI image generation failed: {e}")
            return None

    def _package_images(self, images, params):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for i, img in enumerate(images):
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                zipf.writestr(f'image_{i}.png', img_bytes.getvalue())
            
            metadata = {
                'topic': params['topic'],
                'count': len(images),
                'generation_date': datetime.now().isoformat(),
                'sources': ['web']*(params['size']-len(images)) + ['ai']*min(len(images), params['size'])
            }
            zipf.writestr('metadata.json', json.dumps(metadata))
        
        zip_buffer.seek(0)
        return zip_buffer

    def _parse_resolution(self, res_str: str) -> Tuple[int, int]:
        try:
            width, height = map(int, res_str.lower().split('x'))
            return max(64, min(width, 2048)), max(64, min(height, 2048))
        except:
            print(f"Invalid resolution format '{res_str}'. Using default 512x512")
            return (512, 512)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Dataset Generator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .hero-section {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            padding: 4rem 0;
            border-radius: 0 0 20px 20px;
            margin-bottom: 3rem;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            margin-bottom: 20px;
        }
        .card:hover { transform: translateY(-5px); }
        .form-container {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .btn-primary { background-color: #6e8efb; border-color: #6e8efb; }
        .btn-primary:hover { background-color: #5a7afa; border-color: #5a7afa; }
        .dataset-params { display: none; }
        #downloadSection { display: none; }
        .spinner-border { vertical-align: middle; margin-left: 5px; }
    </style>
</head>
<body>
    <div class="hero-section text-center">
        <div class="container">
            <h1 class="display-4 fw-bold mb-4">Advanced Dataset Generator</h1>
            <p class="lead">Create custom datasets for your projects in seconds</p>
        </div>
    </div>

    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="form-container">
                    <h2 class="mb-4 text-center">Generate Your Dataset</h2>
                    
                    <form id="datasetForm">
                        <div class="mb-4">
                            <label for="dataType" class="form-label fw-bold">What type of data do you need?</label>
                            <select class="form-select" id="dataType" name="data_type" required>
                                <option value="" selected disabled>Select dataset type</option>
                                <option value="tabular">Tabular Data (CSV)</option>
                                <option value="time_series">Time Series Data</option>
                                <option value="text">Text Data (JSON)</option>
                                <option value="image">Image Dataset (ZIP)</option>
                                <option value="graph">Graph/Network Data</option>
                            </select>
                        </div>
                        
                        <div id="parametersContainer">
                            <div id="structuredParams" class="dataset-params">
                                <div class="mb-3">
                                    <label for="topic" class="form-label">Dataset Topic</label>
                                    <input type="text" class="form-control" id="topic" name="topic" placeholder="e.g., healthcare metrics" required>
                                </div>
                                <div class="mb-3">
                                    <div class="form-check form-switch mb-2">
                                        <input class="form-check-input" type="checkbox" id="customFields" name="custom_fields">
                                        <label class="form-check-label" for="customFields">Use custom fields</label>
                                    </div>
                                    <div id="defaultFieldsSection">
                                        <label class="form-label">Default Fields</label>
                                        <input type="text" class="form-control" name="default_fields" value="id, feature1, feature2, target" readonly>
                                    </div>
                                    <div id="customFieldsSection" style="display: none;">
                                        <label for="fieldList" class="form-label">Custom Fields (comma separated)</label>
                                        <input type="text" class="form-control" id="fieldList" name="field_list" placeholder="e.g., id, name, age, score">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="textParams" class="dataset-params">
                                <div class="mb-3">
                                    <label for="textTopic" class="form-label">Text Content Topic</label>
                                    <input type="text" class="form-control" id="textTopic" name="topic" placeholder="e.g., product reviews" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Fields</label>
                                    <input type="text" class="form-control" name="default_fields" value="id, text, sentiment" readonly>
                                </div>
                            </div>
                            
                            <div id="imageParams" class="dataset-params">
                                <div class="mb-3">
                                    <label for="imageTopic" class="form-label">Image Description</label>
                                    <input type="text" class="form-control" id="imageTopic" name="topic" placeholder="e.g., cats, landscapes" required>
                                </div>
                                <div class="mb-3">
                                    <label for="resolution" class="form-label">Resolution</label>
                                    <input type="text" class="form-control" id="resolution" name="resolution" value="512x512">
                                </div>
                                <div class="mb-3">
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="preferAi" name="prefer_ai" checked>
                                        <label class="form-check-label" for="preferAi">Prefer AI-generated images</label>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="safesearch" name="safesearch" checked>
                                        <label class="form-check-label" for="safesearch">Safe search</label>
                                    </div>
                                </div>
                            </div>
                            
                            <div id="graphParams" class="dataset-params">
                                <div class="mb-3">
                                    <label for="graphTopic" class="form-label">Graph Description</label>
                                    <input type="text" class="form-control" id="graphTopic" name="topic" placeholder="e.g., social network" required>
                                </div>
                                <div class="mb-3">
                                    <div class="form-check form-switch mb-2">
                                        <input class="form-check-input" type="checkbox" id="customGraphFields" name="custom_fields">
                                        <label class="form-check-label" for="customGraphFields">Use custom node attributes</label>
                                    </div>
                                    <div id="defaultGraphFieldsSection">
                                        <label class="form-label">Default Attributes</label>
                                        <input type="text" class="form-control" name="default_fields" value="id, community, value" readonly>
                                    </div>
                                    <div id="customGraphFieldsSection" style="display: none;">
                                        <label for="graphFieldList" class="form-label">Custom Attributes (comma separated, must include 'id')</label>
                                        <input type="text" class="form-control" id="graphFieldList" name="field_list" placeholder="e.g., id, group, weight">
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label for="graphSize" class="form-label">Number of Nodes</label>
                                    <input type="number" class="form-control" id="graphSize" name="size" min="10" max="1000" value="50" required>
                                </div>
                                <div class="mb-3">
                                    <label for="edgeProbability" class="form-label">Edge Probability (0.01-0.5)</label>
                                    <input type="number" class="form-control" id="edgeProbability" name="edge_probability" min="0.01" max="0.5" step="0.01" value="0.1" required>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="size" class="form-label">Number of Samples</label>
                            <input type="number" class="form-control" id="size" name="size" min="1" max="10000" value="100" required>
                        </div>
                        
                        <div class="d-grid gap-2 mt-4">
                            <button type="submit" class="btn btn-primary btn-lg" id="generateBtn">
                                <span id="generateText">Generate Dataset</span>
                                <span id="spinner" class="spinner-border spinner-border-sm" role="status" aria-hidden="true" style="display: none;"></span>
                            </button>
                        </div>
                    </form>
                    
                    <div id="downloadSection" class="mt-4 text-center">
                        <h4>Your dataset is ready!</h4>
                        <a id="downloadLink" href="#" class="btn btn-success btn-lg mt-2">
                            <i class="bi bi-download"></i> Download Dataset
                        </a>
                        <div id="errorSection" class="alert alert-danger mt-3" style="display: none;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dataTypeSelect = document.getElementById('dataType');
            const customFieldsCheckbox = document.getElementById('customFields');
            const customGraphFieldsCheckbox = document.getElementById('customGraphFields');
            
            // Initialize form based on default selection
            updateFormVisibility();
            
            dataTypeSelect.addEventListener('change', updateFormVisibility);
            
            if (customFieldsCheckbox) {
                customFieldsCheckbox.addEventListener('change', function() {
                    document.getElementById('defaultFieldsSection').style.display = this.checked ? 'none' : 'block';
                    document.getElementById('customFieldsSection').style.display = this.checked ? 'block' : 'none';
                });
            }
            
            if (customGraphFieldsCheckbox) {
                customGraphFieldsCheckbox.addEventListener('change', function() {
                    document.getElementById('defaultGraphFieldsSection').style.display = this.checked ? 'none' : 'block';
                    document.getElementById('customGraphFieldsSection').style.display = this.checked ? 'block' : 'none';
                });
            }
            
            document.getElementById('datasetForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const generateBtn = document.getElementById('generateBtn');
                const generateText = document.getElementById('generateText');
                const spinner = document.getElementById('spinner');
                const downloadSection = document.getElementById('downloadSection');
                const errorSection = document.getElementById('errorSection');
                
                // Reset UI
                errorSection.style.display = 'none';
                downloadSection.style.display = 'none';
                
                // Show loading state
                generateBtn.disabled = true;
                generateText.textContent = 'Generating...';
                spinner.style.display = 'inline-block';
                
                fetch('/generate_dataset', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.message || 'Failed to generate dataset');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    const downloadLink = document.getElementById('downloadLink');
                    downloadLink.href = data.download_url;
                    downloadLink.textContent = `Download ${data.filename}`;
                    downloadSection.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    errorSection.textContent = error.message || 'An error occurred while generating the dataset.';
                    errorSection.style.display = 'block';
                })
                .finally(() => {
                    generateBtn.disabled = false;
                    generateText.textContent = 'Generate Dataset';
                    spinner.style.display = 'none';
                    
                    if (errorSection.style.display === 'none') {
                        downloadSection.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
            
            function updateFormVisibility() {
                const dataType = dataTypeSelect.value;
                document.querySelectorAll('.dataset-params').forEach(el => {
                    el.style.display = 'none';
                });
                
                if (dataType === 'tabular' || dataType === 'time_series') {
                    document.getElementById('structuredParams').style.display = 'block';
                } else if (dataType === 'text') {
                    document.getElementById('textParams').style.display = 'block';
                } else if (dataType === 'image') {
                    document.getElementById('imageParams').style.display = 'block';
                } else if (dataType === 'graph') {
                    document.getElementById('graphParams').style.display = 'block';
                }
                
                document.getElementById('downloadSection').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    try:
        params = request.form.to_dict()
        params['size'] = int(params.get('size', 100))
        
        # Convert checkbox values to boolean
        params['prefer_ai'] = 'prefer_ai' in params
        params['safesearch'] = 'safesearch' in params
        params['custom_fields'] = 'custom_fields' in params
        
        # Handle fields for structured data
        if params['data_type'] in ['tabular', 'time_series', 'graph']:
            if params.get('custom_fields'):
                params['fields'] = [f.strip() for f in params.get('field_list', '').split(',') if f.strip()]
            else:
                params['fields'] = [f.strip() for f in params.get('default_fields', '').split(',') if f.strip()]
        
        generator = DatasetGenerator()
        data = generator.generate_data(params)
        
        # Save the file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = f"{params['topic'].replace(' ', '_')}_{params['data_type']}_{timestamp}"
        filename = ""
        
        if params['data_type'] == 'image':
            filename = f"{basename}.zip"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(data.getvalue())
        elif params['data_type'] in ['tabular', 'time_series']:
            filename = f"{basename}.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data.to_csv(filepath, index=False)
        elif params['data_type'] == 'text':
            filename = f"{basename}.json"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data.to_json(filepath, orient='records')
        elif params['data_type'] == 'graph':
            filename = f"{basename}.zip"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            generator._save_graph_data(data, os.path.join(app.config['UPLOAD_FOLDER'], basename))
        
        return jsonify({
            'filename': filename,
            'download_url': f'/download/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True
        )
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
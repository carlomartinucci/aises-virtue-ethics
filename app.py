from flask import Flask, abort, jsonify
import os
import logging
import csv

"""
Simple webapp to show scenarios, responses, critics, ratings to responses.

To run:

```
pip install flask
export FLASK_DEBUG=1 && flask run
```

Then open http://127.0.0.1:5000/ in your browser.
"""

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_rating(scenario_name, model_name, content_type):
    """Get rating from CSV if it exists for this scenario and model."""
    try:
        # Extract step number from content_type (e.g., 'wwyd1' -> '1')
        step = content_type.replace('wwyd', '') if content_type.startswith('wwyd') else ''
        if not step:  # If it's just 'wwyd', use step 0
            step = '0'
            
        # Construct the column name (e.g., 'gpt-3.5-turbo-0125_step1')
        column_name = f"{model_name}-step{step}"
        
        with open('rate_answers/ratings.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['scenario'] == f"{scenario_name}.txt":
                    rating = row.get(column_name)
                    if rating and rating.strip():  # Check if rating exists and is not empty
                        return int(rating)
    except Exception as e:
        logger.error(f"Error reading ratings CSV: {str(e)}")
    return None

@app.route("/")
def index():
    html_sections = []
    base_dir = "scenario"
    
    # Get all subdirectories in the scenario folder
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            scenarios = []
            # Get all .txt files from the current subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith(".txt"):
                    # Exclude files that appear to be variants (e.g., name-wwyd.txt) from the main list
                    if '-' not in filename.split('.')[0] or filename.split('.')[0].rsplit('-', 1)[-1] not in ["wwyd", "wwyd1", "wwyd2", "critic", "critic1"]:
                        name = filename.replace(".txt", "").replace("-", " ").title()
                        url_path = f"/scenario/{subdir}/{filename.replace('.txt', '')}"
                        scenarios.append(f'<li><a href="{url_path}">{name}</a></li>')
            
            if scenarios:
                scenario_list = "\n".join(sorted(scenarios)) # Sort scenarios alphabetically
                section_title = subdir.replace("-", " ").title()
                html_sections.append(f"""
                <section>
                    <h2>{section_title}</h2>
                    <ul>
                        {scenario_list}
                    </ul>
                </section>
                """)
    
    content = "\n".join(html_sections)
    
    return f"""
    <html>
        <head>
            <title>Ethics Scenarios</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #444; margin-top: 30px; }}
                section {{ margin-bottom: 40px; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin: 10px 0; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Ethics Scenarios</h1>
            {content}
        </body>
    </html>
    """

@app.route("/scenario/<subdir>/<scenario_base_name>")
def show_scenario(subdir, scenario_base_name):
    logger.debug(f"Requested scenario: subdir={subdir}, scenario_base_name={scenario_base_name}")
    
    main_file_path = os.path.join("scenario", subdir, f"{scenario_base_name}.txt")
    logger.debug(f"Looking for main file at: {main_file_path}")
    
    if not os.path.exists(main_file_path):
        logger.error(f"Main file not found: {main_file_path}")
        abort(404)
    
    try:
        with open(main_file_path, 'r') as f:
            main_content = f.read()
        logger.debug(f"Successfully read main file: {main_file_path}")
    except Exception as e:
        logger.error(f"Error reading main file: {str(e)}")
        abort(500)
    
    title = scenario_base_name.replace("-", " ").title()
    variant_types = ["wwyd", "wwyd1", "wwyd2", "critic", "critic1"]

    return f"""
    <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .main-scenario-content, .variant-content {{ white-space: pre-wrap; line-height: 1.6; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; border-radius: 4px; min-height: 100px;}}
                .columns-container {{ display: flex; justify-content: space-between; gap: 20px; margin-top: 30px; }}
                .column {{ flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; background-color: #fff; }}
                .column h2 {{ margin-top: 0; }}
                .back-link {{ margin-bottom: 20px; display: inline-block; }}
                select, input[type='text'] {{ padding: 8px; margin-bottom: 10px; width: calc(100% - 18px); border-radius: 4px; border: 1px solid #ccc; }}
                .rating {{ font-weight: bold; color: #0066cc; margin-top: 10px; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="back-link">
                <a href="/">← Back to Scenarios List</a>
            </div>
            <h1>{title}</h1>
            <h2>Main Scenario</h2>
            <div class="main-scenario-content">
                {main_content}
            </div>

            <div class="columns-container">
                <div class="column" id="column-1">
                    <h2>Column 1</h2>
                    <label for="variant-select-1">Content Type:</label>
                    <select id="variant-select-1" onchange="loadVariantContent(1, '{subdir}', '{scenario_base_name}')">
                        <option value="">Select Content Type...</option>
                        {''.join([f'<option value="{vt}">{vt.replace("-", " ").title()}</option>' for vt in variant_types])}
                    </select>
                    <label for="model-input-1">Model Name:</label>
                    <input type="text" id="model-input-1" value="gpt-3.5-turbo-0125" placeholder="Enter Model Name" onchange="loadVariantContent(1, '{subdir}', '{scenario_base_name}')">
                    <div class="variant-content" id="variant-content-1">Select a content type and specify model to load.</div>
                    <div class="rating" id="rating-1"></div>
                </div>
                <div class="column" id="column-2">
                    <h2>Column 2</h2>
                    <label for="variant-select-2">Content Type:</label>
                    <select id="variant-select-2" onchange="loadVariantContent(2, '{subdir}', '{scenario_base_name}')">
                        <option value="">Select Content Type...</option>
                        {''.join([f'<option value="{vt}">{vt.replace("-", " ").title()}</option>' for vt in variant_types])}
                    </select>
                    <label for="model-input-2">Model Name:</label>
                    <input type="text" id="model-input-2" value="gpt-3.5-turbo-0125" placeholder="Enter Model Name" onchange="loadVariantContent(2, '{subdir}', '{scenario_base_name}')">
                    <div class="variant-content" id="variant-content-2">Select a content type and specify model to load.</div>
                    <div class="rating" id="rating-2"></div>
                </div>
            </div>

            <script>
                async function loadVariantContent(columnNum, subdir, scenarioBaseName) {{
                    const variantTypeSelect = document.getElementById(`variant-select-${{columnNum}}`);
                    const modelInput = document.getElementById(`model-input-${{columnNum}}`);
                    const contentElement = document.getElementById(`variant-content-${{columnNum}}`);
                    const ratingElement = document.getElementById(`rating-${{columnNum}}`);
                    
                    const variantType = variantTypeSelect.value;
                    const modelName = modelInput.value;

                    if (!variantType) {{
                        contentElement.innerHTML = 'Please select a Content Type.';
                        ratingElement.innerHTML = '';
                        return;
                    }}
                    if (!modelName) {{
                        contentElement.innerHTML = 'Please enter a Model Name.';
                        ratingElement.innerHTML = '';
                        return;
                    }}

                    contentElement.innerHTML = 'Loading...';
                    ratingElement.innerHTML = '';
                    try {{
                        const response = await fetch(`/scenario_content/${{subdir}}/${{scenarioBaseName}}/${{variantType}}/${{modelName}}`);
                        if (response.ok) {{
                            const data = await response.json();
                            if (data.error) {{
                                contentElement.innerHTML = `<p style=\\"color:red;\\">${{data.error}}</p>`;
                                ratingElement.innerHTML = '';
                            }} else {{
                                contentElement.innerHTML = data.content;
                                // Display rating if it exists and this is a wwyd variant
                                if (variantType.startsWith('wwyd') && data.rating !== null) {{
                                    ratingElement.innerHTML = `Rating: ${{data.rating}}`;
                                }} else {{
                                    ratingElement.innerHTML = '';
                                }}
                            }}
                        }} else {{
                            contentElement.innerHTML = `<p style=\\"color:red;\\">Error: ${{response.status}} - ${{response.statusText}}</p><p>File Path Checked (Server-Side): ${{variantType}}/${{subdir}}/${{modelName}}/${{scenarioBaseName}}.txt</p>`;
                            ratingElement.innerHTML = '';
                        }}
                    }} catch (error) {{
                        contentElement.innerHTML = `<p style=\\"color:red;\\">Network or other error fetching content: ${{error}}</p>`;
                        ratingElement.innerHTML = '';
                    }}
                }}
            </script>
        </body>
    </html>
    """

@app.route("/scenario_content/<subdir>/<scenario_base_name>/<variant_type>/<model_name>")
def get_scenario_content(subdir, scenario_base_name, variant_type, model_name):
    logger.debug(f"Requested content: subdir(scenario_type)='{subdir}', scenario_base_name='{scenario_base_name}', variant_type(content_type)='{variant_type}', model_name='{model_name}'")
    
    file_path = os.path.join(variant_type, subdir, model_name, f"{scenario_base_name}.txt")
    logger.debug(f"Attempting to read file at: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Successfully read content file: {file_path}")
        
        # Check for rating if this is a wwyd variant
        rating = None
        if variant_type.startswith('wwyd'):
            rating = get_rating(scenario_base_name, model_name, variant_type)
            
        return jsonify({
            "content": content,
            "rating": rating
        })
    except Exception as e:
        logger.error(f"Error reading content file {file_path}: {str(e)}")
        return jsonify({"error": "Error reading content file."}), 500

# Ensure the logger is available for gunicorn or other WSGI servers
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers.extend(gunicorn_logger.handlers)
    app.logger.setLevel(gunicorn_logger.level)
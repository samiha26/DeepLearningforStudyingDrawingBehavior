import sys, os
import base64
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask import flash

# # Get the parent directory
# parent_dir = os.path.dirname(os.path.realpath(__file__))[:-8]

# Add the parent directory to sys.path
# print(parent_dir)
# print(os.curdir)
sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior")
sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb")
sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model_loaders")

# sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\house")
# sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\tree")
# sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\person")

# sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict")

print(*sys.path, sep='\n')

from dfsdb.src import appWeb

UPLOAD_FOLDER = """D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict"""

app = Flask(__name__)
app.secret_key ='jerry'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/draw-desc.html')
def draw_desc():
    return render_template('draw-desc.html')

@app.route('/draw-house.html')
def draw_house():
    return render_template('draw-house.html')

@app.route('/draw-person.html')
def draw_person():
    return render_template('draw-person.html')

@app.route('/draw-tree.html')
def draw_tree():
    return render_template('draw-tree.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.route('/upload.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Mapping between input names and fixed filenames
        file_names = {
            'house': 'predictHouse.png',
            'tree': 'predictTree.png',
            'person': 'predictPerson.png'
        }
        for input_name, predict_filename in file_names.items():
            file = request.files.get(input_name)
            if file:
                # Check if the uploaded file has an allowed extension
                if file.filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
                    flash('Error: Invalid file format. Please upload a PNG, JPG, or JPEG file.')
                    return redirect(request.url)  # Redirect back to the upload page
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], predict_filename)
                file.save(file_path)
        return redirect(url_for('result_house'))  # Redirect to the result page after upload
    return render_template('upload.html')

# new updated save image
@app.route('/save_drawing', methods=['POST'])
def save_drawing():
    data = request.get_json()
    image_data = data.get("image")
    if image_data:
        # Decode the base64-encoded image data
        _, encoded_data = image_data.split(",", 1)
        decoded_data = base64.b64decode(encoded_data)
        
        current_page = request.referrer
        if current_page.endswith("/draw-house.html"):
            file_name = "predictHouse.png"
        elif current_page.endswith("/draw-tree.html"):
            file_name = "predictTree.png"
        elif current_page.endswith("/draw-person.html"):
            file_name = "predictPerson.png"
        else:
            file_name = "drawing.png"

        # Specify the file path to save the image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

        # Save the image to the specified directory
        with open(file_path, "wb") as f:
            f.write(decoded_data)

        # Determine the next page based on the current page
        current_page = request.referrer  # Get the URL of the current page
        if file_name == "predictHouse.png":
            next_page = "/draw-tree.html"
        elif file_name == "predictTree.png":
            next_page = "/draw-person.html"
        else:
            next_page = "/result-house.html"
        # Construct the response with the URL of the next page
        response = {"message": "Drawing saved successfully!", "nextPage": next_page}
        return jsonify(response), 200
    else:
        return jsonify({"error": "No drawing data received"}), 400

results = appWeb.Results()

@app.route('/result-house.html')
def result_house():
    result = results.get_house_result()
    return render_template('result-house.html', result_item = "House", result = result)

@app.route('/result-person.html')
def result_person():
    result = results.get_person_result()
    return render_template('result-person.html', result_item = "Person", result = result)

@app.route('/result-tree.html')
def result_tree():
    result = results.get_tree_result()
    return render_template('result-tree.html', result_item = "Tree", result = result)

@app.route('/result-overall.html')
def result_overall():
    result = results.overall_result()
    return render_template('result-overall.html', result_item = "Overall", result = result)

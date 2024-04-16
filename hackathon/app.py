from flask import Flask,render_template,request,send_from_directory
from PIL import Image
import io
import os
from prediction_algo import *
import matplotlib
matplotlib.use('agg')
from flask import jsonify

app = Flask(__name__, static_url_path='/static')  # Set the URL path for static files
output_images_folder = os.path.join(app.root_path, 'output images')  # Path to the output images folder


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/model')
def model():
    product_description = "This is the product description from the backend."
    return render_template('upload.html',product_description=product_description)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'upload_file' in request.files:
        file = request.files['upload_file']
        print(file)
        if file.filename != '':
            # Access the file data without saving
            file_data = file.read()
            pil_image = Image.open(io.BytesIO(file_data))
            output = io.BytesIO()
            jpeg_data = output.getvalue()
        
            pil_image.save("output images/output.jpg", format='JPEG')

            image_and_model = gis(image_path=os.path.join(output_images_folder, 'output.jpg'), model_path="verison3/last.pt")

            trees_area, building_area, extra_area = image_and_model.pred_plotting(scale = 100, figsize = (10,10))
            image_and_model.comparision_plot()
            print(trees_area, building_area, extra_area)


            tree_count = trees_area/0.4
            before = round((tree_count * 35) // 1000, 2)

            tree_count = int(trees_area + extra_area) /0.4
            after = round((tree_count * 35) // 1000, 2)

            percent = round((before / after) * 100, 2)
            sugestion = f"Before utilising extra space the Carbon absorption rate is {before} tonnes per year\nAfter utilising extra space the Carbon absorption rate is {after} tonnes per year\nIf Utilised there will be a hike of {percent}% in Carbon Absorption rate"
            print(sugestion)

            response_data = {'sugestion':sugestion}
            
            return jsonify(response_data)
    return 'No file uploaded.'


@app.route('/output_images/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(output_images_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
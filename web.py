from flask import Flask, render_template, request, send_file
import os
from PIL import Image
from ultralytics import YOLO
import openai
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CONFIDENCE_THRESHOLD = 0.6  
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        image = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        
        detected_objects = detect_objects(image_path)
        
        
        caption = generate_image_caption(image_path)
        
        
        description = generate_advanced_description(caption)

        return render_template('result.html', 
                               image_filename=image.filename, 
                               detected_objects=detected_objects, 
                               caption=caption, 
                               description=description)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


def detect_objects(image_path):
    model = YOLO('yolov8x.pt')  
    results = model(image_path)

    detected_objects = {}
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence >= CONFIDENCE_THRESHOLD:
                cls_name = model.names[int(box.cls[0])] 
                if cls_name in detected_objects:
                    detected_objects[cls_name] += 1 
                else:
                    detected_objects[cls_name] = 1 

    if detected_objects:
        
        result_image_path = os.path.join(PROCESSED_FOLDER, 'yolo_' + os.path.basename(image_path))
        results[0].plot()  
        results[0].save(result_image_path)  

        return detected_objects
    else:
        return {"No reliable objects detected": 0}


def generate_image_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_advanced_description(caption):
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                f"Caption: {caption}\n\n"
                "Provide a clear and concise description of the scene mentioned in the caption. Stick strictly to the elements presented without adding extra details or imaginative elements. "
                "Format the description with appropriate indentations and line breaks for readability."
            )}
        ]
    )
    description = completion.choices[0].message['content'].strip()
    return description

if __name__ == '__main__':
    app.run(debug=True)

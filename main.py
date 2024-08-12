import os
from PIL import Image
from ultralytics import YOLO
import openai
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_objects(image_path):
    """Detect objects in the provided image using YOLOv8 and save the image with bounding boxes."""
    print("Loading YOLOv8 model for object detection...")
    model = YOLO('yolov8x.pt')  
    
    print(f"Running object detection on {image_path}...")
    results = model(image_path)

    
    for result in results:
        result.plot()
        result.save("yolo.png")  

    
    detected_objects = {}
    for result in results:
        for box in result.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in detected_objects:
                detected_objects[cls_name] += 1
            else:
                detected_objects[cls_name] = 1

    print(f"Objects detected: {detected_objects}")
    return detected_objects

def generate_image_caption(image_path):
    """Generate a caption for the provided image using the BLIP model."""
    print("Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    print("Loading and processing image...")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    print("Generating caption from image...")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    print(f"Caption generated: {caption}")
    return caption

def generate_advanced_description(caption):
    """Generate an advanced, multi-line description using GPT-4o-mini."""
    print("Sending request to OpenAI API for detailed description...")
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
    print("Description generated successfully.")
    return description

def save_description_to_file(detected_objects, caption, description, output_file):
    """Save the detected objects, caption, and description to a file."""
    print(f"Saving detected objects and description to {output_file}...")
    with open(output_file, "w") as f:
        
        f.write("Detected objects:\n")
        for obj, count in detected_objects.items():
            f.write(f"{obj}: {count}\n")

        
        f.write(f"\nCaption: {caption}\n\n")
        f.write(f"Description:\n{description}\n")

    print("Data saved successfully.")

def main():
    """Process the input image and save the refined description to a text file."""
    image_path = "input.jpeg"
    output_file = "image_description.txt"

    print("Starting object detection process...")
    detected_objects = detect_objects(image_path)
    
    print("Starting image captioning process...")
    caption = generate_image_caption(image_path)
    
    print("Starting detailed description generation process...")
    description = generate_advanced_description(caption)
    
    
    save_description_to_file(detected_objects, caption, description, output_file)
    
    print("Process completed successfully.")

if __name__ == "__main__":
    main()

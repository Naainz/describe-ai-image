import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def generate_image_description(image_path):
    """Generate a description for the provided image using a pre-trained model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    
    return description

def main():
    """Process the input image and save the description to a text file."""
    image_path = "input.jpeg"
    output_file = "image_description.txt"

    description = generate_image_description(image_path)
    with open(output_file, "w") as f:
        f.write(f"{description}\n")

if __name__ == "__main__":
    main()

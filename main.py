import os
import openai
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_image_caption(image_path):
    """Generate a caption for the provided image using the BLIP model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

def generate_advanced_description(caption):
    """Generate an advanced, multi-line description using GPT-4o-mini."""
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                f"Caption: {caption}\n\n"
                "Generate a detailed description of the scene described in the caption. Focus on the elements mentioned. "
                "Don't include arbitrary or unrelated information."
            )}
        ]
    )

    return completion.choices[0].message['content'].strip()

def main():
    """Process the input image and save the refined description to a text file."""
    image_path = "input.jpeg"
    output_file = "image_description.txt"

    
    caption = generate_image_caption(image_path)
    
    
    description = generate_advanced_description(caption)
    
    
    with open(output_file, "w") as f:
        f.write(f"Caption: {caption}\n\n")
        f.write(f"Description:\n{description}\n")

if __name__ == "__main__":
    main()

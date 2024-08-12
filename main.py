import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image

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
    """Generate an advanced, multi-line description using DistilGPT-2."""
    
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    
    prompt = (
        f"Here is a caption: {caption}. "
        "Please generate a detailed description of this scene, but do not make any assumptions or add any information that is not explicitly provided in the caption."
    )

    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        temperature=0.7,
        do_sample=True
    )

    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text.strip()

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

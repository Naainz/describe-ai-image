import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration
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
    """Generate an advanced, multi-line description using Pegasus."""
    
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    
    prompt = (
        f"Caption: {caption}\n\n"
        "Generate a detailed description of the caption {caption}. You can include additional information, context, or details that are explicitly mentioned in the caption. Don't include generic or unrelated statements. Be descriptive and provide a detailed account of the caption."
    )

    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=5,  
        no_repeat_ngram_size=3,
        temperature=0.6,  
        top_p=0.9,  
        early_stopping=True,
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

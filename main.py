import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import spacy


nlp = spacy.load("en_core_web_sm")

def extract_tags(text):
    """Extract relevant tags (lemmas) from the provided text."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

def create_image_description(image_path):
    """Generate a simple and generic description of the image."""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    tags = extract_tags(filename)

    if tags:
        description = f"This image captures elements of {', '.join(tags)}."
    else:
        description = "This image presents a captivating visual."

    return description

def main():
    """Process the input image and save the description to a text file."""
    image_path = "input.jpeg"
    output_file = "image_description.txt"

    description = create_image_description(image_path)
    with open(output_file, "w") as f:
        f.write(f"{os.path.basename(image_path)}: {description}\n")

if __name__ == "__main__":
    main()

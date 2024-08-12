# Describe images with AI

## Description

This project shows how to use [YOLOv8](https://github.com/ultralytics/ultralytics), [BLIP](https://huggingface.co/docs/transformers/en/model_doc/blip), and [OpenAI GPT-4o-mini](https://openai.com) to describe images with AI.

## Setup

1. Install the required packages:
```bash
pip install flask pillow ultralytics openai transformers python-dotenv
```

2. Create a `.env` file with the following content:
```bash
OPENAI_API_KEY=sk-your-openai-api-key
```

3. Run the script:
```bash
python main.py
```

### Option 2: Run the GUI in a web browser

1. Run the script:
```bash
python web.py
```

2. Open a web browser and go to `http://localhost:5000`

3. Upload an image and follow the provided instructions.

## Results

The script will analyse the image provided and generate a description for it. The script will also run YOLO to detect objects in the image. The script then calls the GPT4o-mini model to generate a description of the image.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
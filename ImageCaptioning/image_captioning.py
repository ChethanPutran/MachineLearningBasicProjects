from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the model from the Hugging face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("pick_ball.jpg")

inputs = processor(image,return_tensors="pt")

# Generate captions
outputs = model.generate(**inputs)

for i,item in enumerate(outputs):
    caption = processor.decode(outputs[0],skip_special_tokens=True)

    print(f"Generated caption {i}:",caption)
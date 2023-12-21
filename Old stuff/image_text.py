from transformers import pipeline



# def intiate_image_to_text_model():
captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
caption = pipeline('image-to-text')
    # return caption


def image_to_caption(img):
    # caption = pipeline('image-to-text')
    image_text= caption(img)[0]
    return image_text
from PIL import Image, ImageOps


def normalize_image(image: Image.Image, size: tuple = (224, 224)) -> Image.Image:
    image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    return image

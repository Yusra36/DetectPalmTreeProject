from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None


class_names =["diseased", "healthy"]

BUCKET_NAME = "date_palm_tree_diseases"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/DatePlamTreeDiseases.h5",
            "/tmp/DatePlamTreeDiseases.h5",
        )
        print("Model loaded successfully.")

        model = tf.keras.models.load_model("/tmp/DatePlamTreeDiseases.h5")

    image = request.files["file"]
    print("Image received successfully.")


    image = np.array(
        Image.open(image).convert("RGB").resize((192, 350)) # image resizing
    )
    print("Image preprocessed successfully.")

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    print("Input Shape:", img_array.shape)
    print("Input Values:", img_array)
    predictions = model.predict(img_array)
    print("Predictions:", predictions)

    try:
        predictions = model.predict(img_array)
    except ValueError as e:
        print("Error during prediction:", e)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    

    return {"class": predicted_class, "confidence": confidence}
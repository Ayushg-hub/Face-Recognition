import tensorflow as tf
import mtcnn
from PIL import Image
import numpy as np

def extract_face(image_path,model_size):
    image = Image.open(image_path)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    detector = mtcnn.MTCNN()

    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x2 = x1 + width
    y2 = y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(model_size)
    image = np.expand_dims(np.asarray(image), axis=0)

    return image

def create_embeddings(model,image):

    image = (image - image.mean())/image.std()
    embeddings = model.predict(image)
    return embeddings

def main():
    filename = "./data/test.jpg"
    model_size = (160, 160)

    face = extract_face(filename,model_size)

    model = tf.keras.models.load_model('./model/facenet_keras.h5')
    model.load_weights('./weights/facenet_keras_weights.h5')

    embeddings = create_embeddings(model,face)

    print(embeddings)




if __name__ == '__main__':
    main()
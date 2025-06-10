import cv2
import numpy as np
import pickle
import faiss
from collections import Counter


def read_actor_labels(file_path='db/actor_labels.pickle'):
    with open(file_path, 'rb') as handle:
        actor_labels = pickle.load(handle)
    
    return actor_labels


def read_faiss_db(db_path='db/embeddings.vector'):
    index = faiss.read_index(db_path)
    return index


def get_input_face_embeddings(frame, app):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    # face search in insightface work worse than in face_recognition
    faces = app.get(rgb_image)
    face_embeddings = [face.embedding.tolist() for face in faces]
    
    return face_embeddings


def manually_most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def get_similar_actor_name(frame, index, actor_labels, app):
    input_face_embeddings = get_input_face_embeddings(frame, app)
    face_names = set()
    for input_face_embedding in input_face_embeddings:
        distances, indexes = index.search(np.array([input_face_embedding]), 3)  # for searching knn was used

        # need to find out optimal threshold
        guessed_actors_id = [index for distance, index in zip(distances[0], indexes[0]) if distance < 500]
        
        if len(guessed_actors_id) != 0:
            guessed_actors_name = [actor_labels.get(label) for label in guessed_actors_id]
            face_names.add(manually_most_common(guessed_actors_name))
    
    return face_names
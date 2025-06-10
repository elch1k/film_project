import cv2
import os
import time
from insightface.app import FaceAnalysis
import faiss
import numpy as np
import pickle


def get_actor_photos_path(path_database: str):
    actor_photos = {}
    for actor in os.listdir(path_database):
        actor_folder = os.path.join(path_database, actor)      
       
        if os.path.isdir(actor_folder):
            photos = []

            for photo in os.listdir(actor_folder):
                photo_path = os.path.join(actor_folder, photo)
                photos.append(photo_path)
            
            actor_photos[actor] = photos
    
    return actor_photos


def get_image_embedding(photo_path, app):
    image = cv2.imread(photo_path)
    if image is None:
        print(f'Can\'t properly read image! SOURCE_PATH: {photo_path}')
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_image)
    if not faces:
        print(f'Can\'t find any face on image! SOURCE_PATH: {photo_path}')
        return None
    elif len(faces) > 1:
        for face in faces:
            if face.det_score > 0.70:
                embedding = face.embedding.tolist()
                return embedding
    else:
        embedding = faces[0].embedding.tolist()
        return embedding


def get_labels_and_embeddings(actor_photos_path, app):
    embeddings = []
    actor_label = {}
    counter = 0
    for actor, photo_paths in actor_photos_path.items():
        for photo_path in photo_paths:
            face_embedding = get_image_embedding(photo_path, app)
            if face_embedding is None:
                continue
            embeddings.append(face_embedding)
            actor_label.update({counter: actor})
            counter+=1
  
    return actor_label, embeddings


def create_face_embeddings_db(embeddings, output_path='vector_db/embeddings.vector'):
    dimension = len(embeddings[0])
    # we also can partitioning the index into Voronoi cells for search optimization
    index = faiss.IndexFlatL2(dimension)  # can be used Inner Product, L2 (Euclidean) distance
    index_with_ids = faiss.IndexIDMap(index)
    vectors = np.array(embeddings).astype(np.float32)
    # adds the vectors to the index with sequential ID’s
    index_with_ids.add_with_ids(vectors, np.arange(len(embeddings)))

    faiss.write_index(index_with_ids, output_path)


def write_actor_labels(actor_labels, output_path='actor_labels.pickle'):
    with open(output_path, 'wb') as handle:
        pickle.dump(actor_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    start_time = time.time()
    path_database = "Face_DB"
    save_path = "actor_face_embeddings.json"
    
    providers = ["CUDAExecutionProvider"]
    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    
    path_of_actor_photos = get_actor_photos_path(path_database)
    actor_labels, embeddings = get_labels_and_embeddings(path_of_actor_photos, app)

    create_face_embeddings_db(embeddings)
    write_actor_labels(actor_labels)

    wasted_time = time.time() - start_time
    
    print(f"{np.round(wasted_time, 2)} seconds was spent...")
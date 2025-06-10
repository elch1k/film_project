import cv2 
import pandas as pd
import time
from faiss_similarity_search import read_actor_labels, read_faiss_db, get_similar_actor_name
from insightface.app import FaceAnalysis


def get_actors_name_from_media(input_data, app):
    actor_labels = read_actor_labels()
    index = read_faiss_db()
    face_names = set()

    # test choice between photo and video
    if ".png" in input_data:
        image = cv2.imread(input_data)
        face_names.update(get_similar_actor_name(image, index, actor_labels, app))

    elif ".mp4" in input_data:  # not effective work when i loop by each frame and try to find for each embeddings
        video = cv2.VideoCapture(input_data)
        fps = video.get(cv2.CAP_PROP_FPS)
        interval = 0.5
        frames_to_skip = int(fps * interval)
        frame_count = 0
        
        
        while True:  # can also use video.isOpened()
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % frames_to_skip == 0:
                face_names.update(get_similar_actor_name(frame, index, actor_labels, app))
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
            
            frame_count+=1
        video.release()

    return face_names
    

def get_possible_movies_name(finded_actors: set, file_path='imdb_films_db/clean_imdb_titles_4_10.csv'):
    matching_films = []
    films_db = pd.read_csv(file_path)

    for row in films_db.itertuples():
        if all(actor in row.cast for actor in finded_actors):
            matching_films.append({
                "film_name": row.original_title_name,
                "imdb_link": row.title_url
            })

    return matching_films


if __name__=="__main__":
    input_data = "test_film_frame/film_part_004.mp4"
    # input_data = "test_film_frame/film_frame_006.png"
    
    start_time = time.time()
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    finded_actors = get_actors_name_from_media(input_data, app)
    matching_films = get_possible_movies_name(finded_actors)
    
    print(finded_actors)
    print(matching_films)
    spend_time = time.time()-start_time
    print(f"{round(spend_time, 2)} seconds was spent...")


    # for 24 seconds video were spent 88.14 seconds with dlib face_recognition & json; 6.02 with insightface on cpu & json; 7.1 with insightface on cpu & faiss
# for 2 seconds video i need like 140.27 to go through all frames and get output video in 30fps (is that much - think so)
# so there is no sense take all frames 4 per seconds will be greate (can even smaller)

# just try to use here simple KNN model may be that will be better to classify and faster calculate? not better
# KDTree, BallTree for efficient nearest neighbor search - also not better
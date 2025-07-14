Movie Intelligence Platform
--

This project encompasses several sub-projects: an IMDb scraper, data analysis of the scraped data, and a film recommendation system that employs both traditional TF-IDF and the more complex SBERT methods. Additionally, it includes an actor face recognition component, with InsightFace serving as the project's initial starting point.

## Project Structure & Components:
+ **IMDb Scraper**: The code for the IMDb scraper is located in the `imdb_scraper` folder. (Note: This feature is currently unavailable due to ongoing repository restructuring.)
+ **Data Analysis:** The data analysis component is available in the `content_analysis` folder.
+ **Film Recommendation System:** The film recommendation system, showcasing implementations of various techniques, can be found in the `film_recommendation_system` folder.
+ **Actor Face Recognition System:** This system is organized within this repository as follows:
  + `Face_DB/`: Contains initial actor faces, organized into subfolders for each individual actor.
  + `db/`: Stores FAISS vectorized data of the actor faces for efficient search.
  + `test_film_frame/`: Contains media data used for testing the algorithm.
  + `create_actor_db.py`: This script converts initial images into vectors and creates the vector database (utilizes CUDA).
  + `faiss_similarity_search.py`: This script performs vector similarity search.
  + `possible_movie_algorithm.py`: This script converts input media into vectors (without CUDA) and attempts to identify films based on actors found in the vector database.

The `imdb_films_db/` folder contains scraped IMDb title data.
Note: *The current project structure is unorganized and lacks a production-ready component.*

You can clone this repository and adapt the provided actor face recognition system's algorithm to create your own vector database, then run `possible_movie_algorithm.py` with your data.

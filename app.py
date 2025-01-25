import json
import os
import requests
import streamlit as st
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from requests import HTTPError
load_dotenv()
es = Elasticsearch(["http://localhost:9200"])


def compute_global_params(index="movies"):
    body = {
        "size": 0,
        "aggs": {
            "avg_vote": {"avg": {"field": "vote_average"}},
            "min_votes": {"percentiles": {"field": "vote_count", "percents": [65]}},
            "max_popularity": {"max": {"field": "popularity"}}
        }
    }
    results = es.search(index=index, body=body)
    C = results['aggregations']['avg_vote']['value']
    m = results['aggregations']['min_votes']['values']['65.0']
    P_max = results['aggregations']['max_popularity']['value']
    return C, m, P_max


C, m, P_max = compute_global_params()
IMAGE_URL = 'https://image.tmdb.org/t/p/w500'
lambda_popularity = 100


def calculate_weighted_rating(v, R, P, C, m, P_max, release_year, current_year, lambda_popularity=0.5, lambda_year=0.1):
    try:
        P = float(P)
    except (TypeError, ValueError):
        print("error p")
        P = 0
    year_normalized = (release_year - 1900) / (current_year - 1900)

    if v > 0:
        wr = ((v / (v + m)) * R) + ((m / (v + m)) * C) + \
            lambda_popularity * (P / P_max if P_max else 0) + \
            lambda_year * year_normalized
    else:
        wr = lambda_popularity * (P / P_max if P_max else 0) + \
            lambda_year * year_normalized
    return wr


def process_recommendations(hits, C, m, P_max, lambda_popularity=0.5, lambda_year=0.1):
    recommendations = []
    for hit in hits:
        rec = hit['_source']
        v = rec.get('vote_count', 0)
        R = rec.get('vote_average', 0)
        P = rec.get('popularity', 0)
        release_year = rec.get('release_year', 0)
        wr = calculate_weighted_rating(
            v, R, P, C, m, P_max, release_year, 2024, lambda_popularity, lambda_year)
        rec['weighted_rating'] = wr
        rec['original_score'] = hit['_score']
        recommendations.append(rec)
    recommendations.sort(key=lambda x: x['weighted_rating'], reverse=True)
    return recommendations


def get_similar(the_id, q="*", num=10, index="movies", vector_field='model_factor', cosine=False):
    response = es.get(index=index, id=the_id)
    src = response['_source']
    if vector_field in src:
        query_vec = src[vector_field]
        q = vector_query(query_vec, vector_field, q=q, cosine=cosine)
        results = es.search(index=index, body=q)
        hits = results['hits']['hits']
        recommendations = process_recommendations(
            hits, C, m, P_max, lambda_popularity)
        return src, recommendations[:num+1]


def vector_query(query_vec, vector_field, q="*", cosine=False):
    if cosine:
        score_fn = "doc['{v}'].size() == 0 ? 0 : cosineSimilarity(params.vector, '{v}') + 1.0"
    else:
        score_fn = "doc['{v}'].size() == 0 ? 0 : sigmoid(1, Math.E, -dotProduct(params.vector, '{v}'))"

    score_fn = score_fn.format(v=vector_field, fn=score_fn)

    return {
        "query": {
            "script_score": {
                "query": {
                    "query_string": {
                        "query": q
                    }
                },
                "script": {
                    "source": score_fn,
                    "params": {
                        "vector": query_vec
                    }
                }
            }
        }
    }


def get_user_recs(the_id, q="*", num=10, users="users", movies="movies", vector_field='model_factor'):
    response = es.get(index=users, id=the_id)
    src = response['_source']
    if vector_field in src:
        query_vec = src[vector_field]
        q = vector_query(query_vec, vector_field, q=q, cosine=False)
        results = es.search(index=movies, body=q)
        hits = results['hits']['hits']
        recommendations = process_recommendations(
            hits, C, m, P_max, lambda_popularity)
        return src, recommendations[:num]


def get_hybrid_recommendations(user_id, content_id, alpha=0.7, num=20,
                               user_index="users", movie_index="movies",
                               user_vector_field='model_factor', content_vector_field='meta_factor'):

    user_src, user_recs = get_user_recs(
        user_id, users=user_index, movies=movie_index, vector_field=user_vector_field, num=num * 2)
    user_scores = {rec['movieId']: rec['original_score'] for rec in user_recs}
    content_src, content_recs = get_similar(
        content_id, index=movie_index, vector_field=content_vector_field, num=num * 2, cosine=False)
    content_scores = {rec['movieId']: rec['original_score']
                      for rec in content_recs}
    combined_scores = {}
    for movie_id in set(user_scores.keys()).union(content_scores.keys()):

        if str(movie_id) == str(content_id):
            continue
        user_score = user_scores.get(movie_id, 0)
        content_score = content_scores.get(movie_id, 0)
        combined_scores[movie_id] = alpha * \
            user_score + (1 - alpha) * content_score
    ranked_movies = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True)[:num]
    recommendations = []
    for movie_id, score in ranked_movies:
        movie_data = es.get(index=movie_index, id=movie_id)['_source']
        movie_data['score'] = score
        recommendations.append(movie_data)
    return recommendations


def search_movies(query, index="movies"):
    response = es.search(
        index=index,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "description"]
                }
            }
        }
    )
    return response["hits"]["hits"]


def get_list_user_id():
    response = es.search(index="users", q="userId:*")

    return map(lambda x: x["_id"], response["hits"]["hits"])


def get_movie_details(movie_id):
    try:
        url = "https://api.themoviedb.org/3/movie/" + \
            str(movie_id) + "?language=en-US"
        APIKEY = os.getenv('API_KEY')
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {APIKEY}"
        }
        movie_info = requests.get(url, headers=headers)
        movie_info = json.loads(movie_info.text)
        return movie_info
    except HTTPError as e:
        if e.response.status_code == 401:
            j = json.loads(e.response.text)
            print(j)


def update_movie_id(movieId):
    st.session_state['movieId'] = movieId


def format_movie_keywords_lists(movie_data):
    movie_data['genres'] = ', '.join(
        [genre['name'] for genre in movie_data.get('genres', [])])
    movie_data['keywords'] = ', '.join(
        [keyword['name'] for keyword in movie_data.get('keywords', [])])
    return movie_data


def main():
    st.set_page_config(layout="wide")
    st.title("Movie Recommendation System")
    st.sidebar.header("Select User")
    user_id = st.sidebar.selectbox("User ID", options=list(
        get_list_user_id()), on_change=clear_search_results)
    st.subheader("Search Movies")
    search_query = st.text_input(
        "Enter movie title or keyword", key="search_query")
    if st.button("Search"):
        search_results = search_movies(search_query.strip())
        if search_results:
            st.write("### Search Results")
            for movie in search_results:
                movie_data = movie["_source"]
                st.write(f"**{movie_data['title']}**")
                st.write(movie_data.get("description",
                         "No description available."))
                poster_path = movie_data.get("poster_path", "")
                if poster_path:
                    st.image(IMAGE_URL + poster_path, width=150)
                st.button(f"View Details - {movie['_id']}", key=f"view-{movie['_id']}", on_click=update_movie_id, args=(movie['_id'],))

        else:
            st.write("No movies found.")
    src, movie_rec_for_user = get_user_recs(user_id)
    if movie_rec_for_user:
        if "movieId" not in st.session_state:
            st.subheader("Movies You May Like")
            show_movie_grid(movie_rec_for_user)
    else:
        st.write("No recommendations available.")
    if "movieId" in st.session_state:
        movie_id = st.session_state["movieId"]
        movie_details = get_movie_details(movie_id)
        movie_details = format_movie_keywords_lists(movie_details)
        if movie_details:
            st.write("## Movie Details")
            st.write(f"**Title:** {movie_details.get('title', 'N/A')}")
            st.write(f"**MovieId** {movie_details.get('id', '')}")
            st.write(
                f"**Genre:** {movie_details.get('genres', 'N/A')}")
            st.write(
                f"**Description:** {movie_details.get('overview', 'No description available.')}")
            st.write(f"**Tagline:** {movie_details.get('tagline', 'N/A')}")
            st.write(
                f"**Release Date:** {movie_details.get('release_date', 'N/A')}")
            st.write(
                f"**Vote Average:** {movie_details.get('vote_average', 'N/A')}")
            st.write(
                f"**Vote Count:** {movie_details.get('vote_count', 'N/A')}")
            st.write(
                f"**Popularity:** {movie_details.get('popularity', 'N/A')}")
            st.write(f"**Keywords:** {movie_details.get('keywords', 'N/A')}")

            poster_path = movie_details.get("poster_path", "")
            if poster_path:
                st.image(IMAGE_URL + poster_path, width=200)

            movie_rec_for_movie = get_hybrid_recommendations(user_id, movie_id)
            st.write("## Recommended Movies")
            if movie_rec_for_movie:
                show_movie_grid(movie_rec_for_movie)
            else:
                st.write("No recommendations available.")


def clear_search_results():
    if "search_query" in st.session_state:
        st.session_state["search_query"] = ""


def show_movie_grid(movies):

    cols = st.columns(4)
    for idx, movie in enumerate(movies):
        movie_id = movie['movieId']
        movie_data = get_movie_details(movie_id)
        if not movie_data:
            continue
        movie_data = format_movie_keywords_lists(movie_data)
        col = cols[idx % 4]
        print(map(lambda x: x['name'], movie_data.get('genres', 'N/A')))
        with col:
            movie_title = movie_data.get("title", "Unknown")
            poster_path = movie_data.get("poster_path", "")
            st.image(IMAGE_URL + poster_path, width=150)
            st.write(f"**{movie_title}**")
            st.write(f"**MovieId** {movie_data.get('id', '')}")
            st.write(
                f"**Genre:** {movie_data.get('genres', 'N/A')}")
            st.write(f"**Tagline:** {movie_data.get('tagline', 'N/A')}")


if __name__ == "__main__":
    main()

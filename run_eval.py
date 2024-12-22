from evaluation import compute_coverage, compute_diversity, compute_novelty
from app import get_hybrid_recommendations  # Import the missing function
import numpy as np
from collections import defaultdict


def get_test_users(es, min_ratings=10):
    query = {
        "size": 0,
        "aggs": {
            "user_ratings": {
                "terms": {
                    "field": "userId",
                    "min_doc_count": min_ratings,
                    "size": 1000
                }
            }
        }
    }
    results = es.search(index="ratings", body=query)
    return [bucket["key"] for bucket in results["aggregations"]["user_ratings"]["buckets"]]


def get_user_test_items(es, user_id, threshold=3.5):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"userId": user_id}},
                    {"range": {"rating": {"gte": threshold}}}
                ]
            }
        }
    }
    response = es.search(index="ratings", body=query)
    return [hit["_source"]["movieId"] for hit in response["hits"]["hits"]]


def get_hybrid_test_recs(es, user_id, movie_id, k=10):
    try:
        recs = get_hybrid_recommendations(user_id, movie_id, num=k)
        return [r["movieId"] for r in recs]
    except:
        return []


def evaluate_hybrid_recommender(es, k=10, test_users=100):
    users = get_test_users(es)[:test_users]
    movies_response = es.search(index="movies", body={"size": 10000})
    movie_ids = [hit["_source"]["movieId"]
                 for hit in movies_response["hits"]["hits"]]
    popularity = {}
    for hit in movies_response["hits"]["hits"]:
        movie = hit["_source"]
        popularity[movie["movieId"]] = movie.get("popularity", 0)
    max_pop = max(popularity.values())
    popularity = {k: float(v) /float(max_pop) for k, v in popularity.items()}
    recommendations = {}
    actual_items = {}

    for user_id in users:
        # Get user's actual highly rated items
        test_items = get_user_test_items(es, user_id)
        if not test_items:
            continue


        seed_movie = test_items[0]
        recs = get_hybrid_test_recs(es, user_id, seed_movie, k)

        if recs:
            recommendations[user_id] = recs
            actual_items[user_id] = set(test_items)

    if not recommendations:
        return {"error": "No recommendations generated"}
    coverage = compute_coverage(recommendations, movie_ids)
    diversity = compute_diversity(recommendations)
    novelty = compute_novelty(recommendations, popularity)
    precisions = []
    recalls = []

    for user_id in recommendations:
        if user_id in actual_items:
            rec_set = set(recommendations[user_id])
            actual_set = actual_items[user_id]

            if len(actual_set) > 0:
                precision = len(rec_set.intersection(
                    actual_set)) / len(rec_set)
                recall = len(rec_set.intersection(
                    actual_set)) / len(actual_set)

                precisions.append(precision)
                recalls.append(recall)

    precision = np.mean(precisions)
    recall = np.mean(recalls)

    return {
        "coverage": coverage,
        "diversity": diversity,
        "novelty": novelty,
        "precision": precision,
        "recall": recall
    }


def print_evaluation_results(results):
    print("\nHybrid Recommender Evaluation Results:")
    print("-------------------------------------")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    from elasticsearch import Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    results = evaluate_hybrid_recommender(es)
    print_evaluation_results(results)
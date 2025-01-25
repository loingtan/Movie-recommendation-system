def compute_coverage(recommendations, item_catalog):
    recommended_items = set(
        item for user_rec in recommendations.values() for item in user_rec)
    coverage = len(recommended_items) / len(item_catalog)
    return coverage


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union
def compute_diversity(recommendations):
    user_item_sets = {user: set(items)
                      for user, items in recommendations.items()}
    diversity_sum = 0
    total_users = len(user_item_sets)
    for user1, items1 in user_item_sets.items():
        for user2, items2 in user_item_sets.items():
            if user1 != user2:
                diversity_sum += jaccard_similarity(items1, items2)
    diversity = diversity_sum / (total_users * (total_users - 1))
    return diversity


def compute_novelty(recommendations, item_popularity):
    total_novelty = 0
    num_users = len(recommendations)
    max_novelty = sum(item_popularity.values())

    for user_rec in recommendations.values():
        novelty_sum = sum(item_popularity[item] for item in user_rec)
        total_novelty += novelty_sum / len(user_rec)
    novelty = total_novelty / (num_users * max_novelty)
    scaled_novelty = novelty * 100  
    return scaled_novelty

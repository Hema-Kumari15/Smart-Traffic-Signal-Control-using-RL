def compute_score(wait, throughput, fairness):
    score = 0
    score += max(0, 1 - wait / 1000)
    score += min(1, throughput / 500)
    score += max(0, 1 - fairness)
    return score / 3

import torch


class ReliabilityTracker:
    def __init__(self):
        self.history = {}

    def compute_reliability(self, client_id, loss, accuracy=None):

        loss_score = 1 / (loss + 1e-8)

        history_score = 1.0
        if client_id in self.history:
            h = self.history[client_id]
            history_score = sum(h[-3:]) / min(len(h), 3)

        score = 0.6 * loss_score + 0.4 * history_score

        if accuracy is not None:
            score = 0.7 * score + 0.3 * accuracy

        score = float(torch.tensor(score).clamp(0, 1))

        if client_id not in self.history:
            self.history[client_id] = []

        self.history[client_id].append(score)

        return score

    def get_client_reliability(self, client_id):
        if client_id not in self.history:
            return 1.0
        return sum(self.history[client_id]) / len(self.history[client_id])

    def get_all_reliabilities(self):
        return {
            k: sum(v) / len(v) for k, v in self.history.items()
        }
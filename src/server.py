import torch


class Server:

    def __init__(self, global_model, reliability_tracker=None):

        self.global_model = global_model

        self.reliability_tracker = reliability_tracker

        self.client_updates = []

        self.client_ids = []

        # Day 11 additions
        self.reliability_history = []

        self.weight_history = []

        self.global_accuracy_history = []

    # -------------------------
    # Receive Client Update
    # -------------------------
    def add_client_update(self, client_id, weights):

        self.client_ids.append(client_id)

        self.client_updates.append(weights)

    # -------------------------
    # Reliability-Aware Aggregation
    # -------------------------
    def aggregate(self):

        if len(self.client_updates) == 0:
            return None

        reliability_weights = []

        for cid in self.client_ids:

            if self.reliability_tracker:

                weight = self.reliability_tracker.get_client_reliability(
                    cid
                )

            else:

                weight = 1.0

            reliability_weights.append(weight)

        # Save reliability history
        self.reliability_history.append(
            reliability_weights.copy()
        )

        total_weight = sum(reliability_weights)

        if total_weight == 0:
            total_weight = 1.0

        reliability_weights = [
            w / total_weight
            for w in reliability_weights
        ]

        # Save aggregation weights
        self.weight_history.append(
            reliability_weights.copy()
        )

        new_weights = {}

        for key in self.client_updates[0].keys():

            first_tensor = self.client_updates[0][key]

            # Integer tensors
            if first_tensor.dtype in (
                torch.int64,
                torch.int32,
                torch.long
            ):

                new_weights[key] = first_tensor.clone()

            else:

                aggregated = torch.zeros_like(
                    first_tensor,
                    dtype=torch.float32
                )

                for i, client_state in enumerate(
                    self.client_updates
                ):

                    aggregated += (
                        client_state[key].float()
                        * reliability_weights[i]
                    )

                new_weights[key] = aggregated

        self.global_model.load_state_dict(
            new_weights
        )

        self.client_updates = []

        self.client_ids = []

        return self.global_model.state_dict()

    # -------------------------
    # Log Global Accuracy
    # -------------------------
    def log_global_accuracy(
        self,
        accuracy
    ):

        self.global_accuracy_history.append(
            accuracy
        )

    # -------------------------
    # Accessors
    # -------------------------
    def get_global_model(self):

        return self.global_model.state_dict()

    def set_global_model(
        self,
        weights
    ):

        self.global_model.load_state_dict(
            weights
        )

    def get_reliability_history(self):

        return self.reliability_history

    def get_weight_history(self):

        return self.weight_history

    def get_accuracy_history(self):

        return self.global_accuracy_history
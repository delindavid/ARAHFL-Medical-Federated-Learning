import torch


class Server:
    def __init__(self, global_model, reliability_tracker=None):
        self.global_model = global_model
        self.reliability_tracker = reliability_tracker

        self.client_updates = []
        self.client_ids = []

    # -------------------------
    # Receive client updates
    # -------------------------
    def add_client_update(self, client_id, weights):
        self.client_ids.append(client_id)
        self.client_updates.append(weights)

    # -------------------------
    # Reliability-Aware Aggregation (ARAHFL)
    # -------------------------
    def aggregate(self):

        if len(self.client_updates) == 0:
            return None

        # Get reliability weights
        reliability_weights = []

        for cid in self.client_ids:
            if self.reliability_tracker:
                w = self.reliability_tracker.get_client_reliability(cid)
            else:
                w = 1.0

            reliability_weights.append(w)

        # Normalize weights
        total_weight = sum(reliability_weights)

        reliability_weights = [
            w / total_weight
            for w in reliability_weights
        ]

        new_weights = {}

        # Aggregate each parameter
        for key in self.client_updates[0].keys():

            first_tensor = self.client_updates[0][key]

            # Handle integer tensors separately
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

                for i, client_state in enumerate(self.client_updates):
                    aggregated += (
                        client_state[key].float()
                        * reliability_weights[i]
                    )

                new_weights[key] = aggregated

        # Update global model
        self.global_model.load_state_dict(new_weights)

        # Clear buffers
        self.client_updates = []
        self.client_ids = []

        return self.global_model.state_dict()

    # -------------------------
    # Get global model
    # -------------------------
    def get_global_model(self):
        return self.global_model.state_dict()

    # -------------------------
    # Set global model
    # -------------------------
    def set_global_model(self, weights):
        self.global_model.load_state_dict(weights)
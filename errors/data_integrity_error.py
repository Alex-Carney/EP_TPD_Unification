class DataIntegrityError(Exception):
    def __init__(self, message: str, experiment_id: str = None):
        self.experiment_id = experiment_id
        full_message = f"[DataIntegrityError] {message}"
        if experiment_id:
            full_message += f" (experiment_id={experiment_id})"
        super().__init__(full_message)

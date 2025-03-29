from models import DQNModel
from utils import load_data

class Client:
    def __init__(self, client_id, data, labels, batch_size=8):
        self.client_id = client_id
        self.model = DQNModel()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def train(self, epochs=3):
        self.model.fit(self.data, self.labels, epochs=epochs, batch_size=self.batch_size, verbose=0)
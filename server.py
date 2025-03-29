import numpy as np
import pandas as pd
from models import DQNModel
from utils import load_data, evaluate_accuracy
import tensorflow as tf

class Server:
    def __init__(self, num_clients=5):
        self.global_model = DQNModel()
        self.num_clients = num_clients
        self.clients = []
        self.global_weights = self.global_model.get_weights()

    def aggregate(self, client_weights):
        new_weights = []
        for weights_list_tuple in zip(*client_weights):
            new_weights.append(np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))
        self.global_model.set_weights(new_weights)

    def register_clients(self, clients):
        self.clients = clients

    def federated_train(self, rounds=10):
        for r in range(rounds):
            client_weights = []
            for client in self.clients:
                client.model.set_weights(self.global_model.get_weights())
                client.train()
                client_weights.append(client.model.get_weights())
            self.aggregate(client_weights)
            print(f"Round {r+1}/{rounds} completed.")

    def evaluate(self, test_data, test_labels):
        accuracy = evaluate_accuracy(self.global_model, test_data, test_labels)
        print("Federated model accuracy:", accuracy)
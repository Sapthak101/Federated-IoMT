from server import Server
from client import Client
from utils import load_data
import numpy as np

X_train, X_test, y_train, y_test = load_data()
num_clients = 5
data_splits = np.array_split(X_train, num_clients)
label_splits = np.array_split(y_train, num_clients)

server = Server(num_clients=num_clients)
clients = [Client(i, data_splits[i], label_splits[i]) for i in range(num_clients)]
server.register_clients(clients)
server.federated_train(rounds=10)
server.evaluate(X_test, y_test)
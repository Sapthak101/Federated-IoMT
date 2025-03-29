import pandas as pd

df = pd.read_csv("synthetic_iomt_dataset.csv")
alpha, beta, gamma = 0.5, 0.3, 0.2

def cost_function(row):
    return alpha * row['execution_time'] + beta * row['communication_cost'] + gamma * row['computation_cost']

df['total_cost'] = df.apply(cost_function, axis=1)
optimal_tasks = df[df['total_cost'] < df['total_cost'].mean()]
print("Optimal task allocation (cost-optimized):")
print(optimal_tasks.head())
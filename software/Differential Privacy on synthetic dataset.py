#Python code to generate a synthetic dataset, consisting of Customer name, Account number, Date of Transaction, Deposit Amount, and apply Differential Privacy (DP)
import pandas as pd
import numpy as np
import random

customer_names = ["Alice", "Bob", "Charlie", "David", "Emily", "Frank", "Grace", "Harry", "Isabella", "Jack"]

account_numbers = [random.randint(10000000, 99999999) for _ in range(len(customer_names))]

transaction_dates = [pd.to_datetime(str(random.randint(2020, 2023)) + "-" + str(random.randint(1, 12)) + "-" + str(random.randint(1, 28))) for _ in range(len(customer_names))]

deposit_amounts = [random.randint(100, 1000) for _ in range(len(customer_names))]

synthetic_data = pd.DataFrame({
    "Customer Name": customer_names,
    "Account Number": account_numbers,
    "Date of Transaction": transaction_dates,
    "Deposit Amount": deposit_amounts
})

def add_noise(data, sigma):
    noise = np.random.normal(0, sigma, size=len(data))
    return data + noise

def compute_epsilon(sigma, delta):
    epsilon = -np.log(delta) / (2 * sigma**2)
    return epsilon

sigma = 10  
delta = 0.01  

synthetic_data["Deposit Amount"] = add_noise(synthetic_data["Deposit Amount"], sigma)
epsilon = compute_epsilon(sigma, delta)

print("Epsilon:", epsilon)


print(synthetic_data)

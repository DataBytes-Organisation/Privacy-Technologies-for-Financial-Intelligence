#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import tenseal as ts
from sympy import Symbol, expand

def create_single_term_polynomial(suspicious_value):
    x = Symbol('x')
    polynomial = (x - suspicious_value)
    print(f"Generated Polynomial: {polynomial}")  # Debug: print the polynomial
    return expand(polynomial)

def evaluate_polynomial(polynomial, value):
    return polynomial.evalf(subs={Symbol('x'): value})

def homomorphic_search_linear(suspicious_list, shared_list):
    # Initialize TenSEAL context with higher scale for better precision
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, 7)
    scale = 2 ** 40

    matches = []

    # Encrypt and compare each shared value to each suspicious value individually
    for shared in shared_list:
        encrypted_shared = ts.ckks_vector(context, [shared], scale)

        for suspicious_value in suspicious_list:
            polynomial = create_single_term_polynomial(suspicious_value)

            result = ts.ckks_vector(context, [0], scale)

            # Evaluate the single-term polynomial (linear) at the encrypted value
            for i, coef in enumerate(polynomial.as_coefficients_dict().values()):
                encrypted_coef = ts.ckks_vector(context, [coef], scale)
                term = encrypted_shared ** i * encrypted_coef
                result += term

            decrypted_result = result.decrypt()[0]

            # Print results for debugging
            direct_result = evaluate_polynomial(polynomial, shared)
            print(f"Evaluating for {shared} with {suspicious_value}: Decrypted Result = {decrypted_result}, Direct Result = {direct_result}")

            # Check if the polynomial evaluates close to zero
            if abs(decrypted_result) < 1e-5:
                matches.append(shared)
                break  # Stop checking once a match is found

    return matches

# Example usage
suspicious_list = np.random.randint(1000000, 9999999, size=5).tolist()  # Reduced list size
shared_list = np.random.randint(1000000, 9999999, size=10).tolist()      # Reduced list size

# Hardcode a common value in both lists
common_value = 1234
suspicious_list.append(common_value)
shared_list.append(common_value)

print(f"Suspicious list: {suspicious_list}")
print(f"Shared list: {shared_list}")

# Call the function with homomorphic encryption and linear validation
matches = homomorphic_search_linear(suspicious_list, shared_list)

# Display matches
print(f"Suspicious people found: {matches}")


# In[ ]:





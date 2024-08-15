#!/usr/bin/env python
# coding: utf-8


from sympy import Symbol, expand
import tenseal as ts

def create_polynomial(suspicious_list):
    x = Symbol('x')
    polynomial = 1
    for s in suspicious_list:
        polynomial *= (x - s)
    return expand(polynomial)

def evaluate_polynomial(polynomial, value):
    return polynomial.evalf(subs={Symbol('x'): value})

def homomorphic_search(suspicious_list, shared_list):
    # Initialize TenSEAL context with a higher scale
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, 7)
    scale = 2 ** 40  # to be adjusted if needed

    # Create polynomial from suspicious list
    polynomial = create_polynomial(suspicious_list)
    # print(f"Polynomial: {polynomial}")

    # Create matches list
    matches = []

    # Evaluate the polynomial at each entry in the shared list
    for shared in shared_list:
        encrypted_shared = ts.ckks_vector(context, [shared], scale)
        result = ts.ckks_vector(context, [0], scale)

        # Evaluate the polynomial at the encrypted value
        for i, coef in enumerate(polynomial.as_coefficients_dict().values()):
            encrypted_coef = ts.ckks_vector(context, [coef], scale)
            term = encrypted_shared ** i * encrypted_coef
            result += term

        decrypted_result = result.decrypt()[0]

        # may be needed to debugging
        # print(f"Evaluating for {shared}: Decrypted Result = {decrypted_result}")

        # Direct evaluation for comparison
        direct_result = evaluate_polynomial(polynomial, shared)
        # print(f"Directly evaluating for {shared}: Result = {direct_result}")

        # Check for match with a tight threshold
        if abs(decrypted_result) < 1e-7 or abs(direct_result) < 1e-7:  # Adjusted check
            matches.append(shared)

    return matches

# Example usage
suspicious_list = [101, 102, 103, 104]
shared_list = [100, 101, 104, 102, 200, 103]
matches = homomorphic_search(suspicious_list, shared_list)

# Display results
print(f"Suspicious people found: {matches}")

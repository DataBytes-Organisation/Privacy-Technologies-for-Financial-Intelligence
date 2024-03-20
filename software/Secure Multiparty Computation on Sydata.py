# Python code for Secure Multiparty Computation (SMC or MPC) using Shamir's Secret Sharing scheme:
import secretsharing

def share_secret(secret, num_parties):
    shares = secretsharing.share(secret, num_parties)
    return shares

def combine_shares(shares):
    secret = secretsharing.unshare(shares)
    return secret

def compute_secure_sum(shares1, shares2):
    combined_shares = []
    for i in range(len(shares1)):
        combined_shares.append(shares1[i] + shares2[i])

    return combined_shares

def test_smc_protocol(secret1, secret2, num_parties):
    shares1 = share_secret(secret1, num_parties)
    shares2 = share_secret(secret2, num_parties)

    # Simulate the SMC protocol among participating parties

    combined_shares = compute_secure_sum(shares1, shares2)
    recovered_sum = combine_shares

    print("Original secrets:", secret1, secret2)
    print("Recovered sum:", recovered_sum)

# Test secure sum with two parties
test_smc_protocol(10, 20, 2)

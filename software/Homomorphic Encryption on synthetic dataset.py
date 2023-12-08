#Python code for Homomorphic Encryption (HE) using the Paillier cryptosystem
import paillier as paillier

def encrypt(message, public_key):
    ciphertext = paillier.encrypt(message, public_key)
    return ciphertext

def decrypt(ciphertext, private_key):
    message = paillier.decrypt(ciphertext, private_key)
    return message

def homomorphic_addition(ciphertext1, ciphertext2, public_key):
    sum_ciphertext = paillier.multiply(ciphertext1, ciphertext2, public_key)
    return sum_ciphertext

def homomorphic_multiplication(ciphertext, scalar, public_key):
    product_ciphertext = paillier.exponentiate(ciphertext, scalar, public_key)
    return product_ciphertext

def test_he_implementation(message1, message2, public_key, private_key):
    ciphertext1 = encrypt(message1, public_key)
    ciphertext2 = encrypt(message2, public_key)

    sum_ciphertext = homomorphic_addition(ciphertext1, ciphertext2, public_key)
    decrypted_sum = decrypt(sum_ciphertext, private_key)

    product_ciphertext = homomorphic_multiplication(ciphertext1, message2, public_key)
    decrypted_product = decrypt(product_ciphertext, private_key)

    print("Original messages:", message1, message2)
    print("Homomorphic addition:", decrypted_sum)
    print("Homomorphic multiplication:", decrypted_product)

# Generate public and private keys
paillier.generate_random_keypair()
public_key = paillier.get_public_key()
private_key = paillier.get_private_key()

# Test homomorphic addition and multiplication
test_he_implementation(10, 20, public_key, private_key)

import base64
import hashlib
import os
from Crypto import Random
from Crypto.Cipher import AES

'''
Thanks to
http://stackoverflow.com/questions/12524994/encrypt-decrypt-using-pycrypto-aes-256
'''
class AESCipher:
    def __init__(self, key):
        self.bs = AES.block_size  # 16 bytes
        self.key = hashlib.sha256(key.encode("utf-8")).digest()[:16] #AES-128

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(self.bs)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        iv = enc[:self.bs]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[self.bs:]))

    def _pad(self, data):
        padding_len = self.bs - len(data) % self.bs
        padding = bytes([padding_len]) * padding_len
        return data + padding

    @staticmethod
    def _unpad(data):
        padding_len = data[-1]
        return data[:-padding_len]

# Generate key 
def generate_encryption_key():
    key = os.urandom(16)  # 16 bytes = 128 bits
    key_b64 = base64.b64encode(key).decode('utf-8')
    return key_b64
import torch
print(torch.__version__)
print("CUDA (GPU) var mı:", torch.cuda.is_available())  # False çıkmalı, CPU kullanıyoruz zaten

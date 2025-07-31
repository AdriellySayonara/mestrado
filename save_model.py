from tensorflow.keras.applications import MobileNetV2

# Carrega o modelo MobileNetV2 com pesos do ImageNet
model = MobileNetV2(weights='imagenet')

# Salva o modelo em um arquivo .h5
model.save('mobilenet_v2_imagenet.h5')

print("Modelo 'mobilenet_v2_imagenet.h5' salvo com sucesso!")
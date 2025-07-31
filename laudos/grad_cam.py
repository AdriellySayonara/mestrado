# laudos/grad_cam.py (Versão Simplificada e Correta)

import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Cria um modelo que mapeia a imagem de entrada para as ativações
    # da última camada conv e para as predições de saída.
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8) # Normaliza
    return heatmap.numpy()

def generate_grad_cam_overlay(img_path, model, preprocessed_img, last_conv_layer_name='mixed10'):
    # A imagem já vem pré-processada para a função de heatmap
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em: {img_path}")
    
    # Obtém o tamanho da entrada do modelo para o redimensionamento
    target_size = (model.input.shape[2], model.input.shape[1])
    original_img = cv2.resize(original_img, target_size)
    
    # Gera o heatmap
    heatmap = make_gradcam_heatmap(preprocessed_img, model, last_conv_layer_name)

    # Redimensiona o heatmap para o tamanho da imagem e aplica o colormap
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    jet_as_img_type = jet.astype(original_img.dtype)
    alpha = 0.4 # Intensidade do heatmap
    superimposed_img = cv2.addWeighted(jet_as_img_type, alpha, original_img, 1 - alpha, 0)
    
    final_image_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    final_image = tf.keras.utils.array_to_img(final_image_rgb)
    
    return final_image
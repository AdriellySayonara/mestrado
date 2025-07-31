# ml_manager/feature_extractor.py
import cv2
import numpy as np
import pywt
import os
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, shannon_entropy

# NOTA: 'segmentation_models' e 'os.environ' foram removidos do topo.

def get_lesion_mask(image_path):
    """
    Usa um modelo U-Net para segmentar a lesão e retornar uma máscara binária.
    A importação e a criação do modelo agora acontecem DENTRO desta função.
    """
    # --- IMPORTAÇÕES E CRIAÇÃO DO MODELO ISOLADAS AQUI ---
    # Define o framework ANTES de importar a biblioteca
    os.environ['SM_FRAMEWORK'] = 'tf.keras'
    import segmentation_models as sm

    BACKBONE = 'resnet34'
    preprocess_input_seg = sm.get_preprocessing(BACKBONE)
    try:
        segmentation_model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')
        print("   ... Modelo de segmentação U-Net (ResNet34) carregado.")
    except Exception as e:
        print(f"   !!! Erro ao carregar o modelo de segmentação: {e} !!!")
        return None
    # ----------------------------------------------------

    IMG_SIZE = (256, 256)
    img = cv2.imread(image_path)
    if img is None:
        print(f"   !!! Aviso: Não foi possível ler a imagem: {image_path} !!!")
        return None
        
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, IMG_SIZE)
    img_preprocessed = preprocess_input_seg(img_resized)
    mask_pred = segmentation_model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)[0]
    mask_binary = (mask_pred > 0.5).astype(np.uint8)
    return cv2.resize(mask_binary, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

def extract_classic_features_from_image(image_path):
    """
    Extrai um vetor de características clássicas, usando a máscara de segmentação.
    """
    image = cv2.imread(image_path)
    if image is None: return np.zeros(14)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask = get_lesion_mask(image_path)
    
    if mask is None or np.sum(mask) == 0:
        print(f"   Aviso: Máscara inválida para {os.path.basename(image_path)}. Retornando zeros.")
        return np.zeros(14)
    
    if mask.ndim == 3: mask = mask[:,:,0]
    
    lesion_texture_only = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    features = []
    
    props = regionprops(mask)
    if props:
        p = props[0]
        features.extend([p.area, p.perimeter, p.eccentricity, p.solidity])
    else:
        features.extend([0, 0, 0, 0])
    
    glcm = graycomatrix(lesion_texture_only, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features.extend([graycoprops(glcm, p)[0, 0] for p in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')])
    
    coeffs = pywt.dwt2(lesion_texture_only, 'haar')
    cA, (cH, cV, cD) = coeffs
    for a in [cA, cH, cV, cD]:
        features.append(np.sum(np.square(a)) / (a.size + 1e-6))
    
    features.append(shannon_entropy(lesion_texture_only))
    return np.array(features)
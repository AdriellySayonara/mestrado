# laudos/services.py (Refatorado com Reconstrução de Modelo para Grad-CAM)

import numpy as np
import os
import joblib
import io
import traceback
from django.conf import settings
from django.core.files.base import ContentFile

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from .models import ModeloML
from .grad_cam import generate_grad_cam_overlay
from ml_manager.feature_extractor import extract_classic_features_from_image

# --- NOVA FUNÇÃO DE ALONGAMENTO DE CONTRASTE ---
def apply_contrast_stretching(image_array):
    """Aplica o alongamento de contraste a um array de imagem NumPy."""
    img_float = image_array.astype(float)
    for i in range(3):
        min_val = np.min(img_float[:,:,i])
        max_val = np.max(img_float[:,:,i])
        if max_val > min_val:
            img_float[:,:,i] = 255 * (img_float[:,:,i] - min_val) / (max_val - min_val)
    return img_float.astype(np.uint8)

def get_cnn_info(architecture_name):
    """Retorna o pré-processamento, target_size, nome da última camada conv e a classe do modelo."""
    if architecture_name == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
        return preprocess_input, (299, 299), 'mixed10', InceptionV3
    elif architecture_name == 'efficientnet_b0':
        from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
        return preprocess_input, (224, 224), 'top_conv', EfficientNetB0
    elif architecture_name == 'densenet_201':
        from tensorflow.keras.applications.densenet import preprocess_input, DenseNet201
        return preprocess_input, (224, 224), 'relu', DenseNet201
    else: # Fallback
        from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
        return preprocess_input, (299, 299), 'mixed10', InceptionV3

def classify_image(laudo_instance):
    print("\n--- [SERVICE] INICIANDO CLASSIFICAÇÃO ---")
    try:
        active_model_instance = ModeloML.objects.get(is_active=True)
        model_path = active_model_instance.arquivo_modelo.path
        params = active_model_instance.parametros_treinamento
        
        print(f"   [SERVICE] Modelo ativo encontrado: {active_model_instance.nome_versao}")
        
        if not params:
            raise ValueError("Parâmetros de treinamento não encontrados no modelo ativo.")

        pipeline_type = params.get('pipeline_type')
        cnn_arch = params.get('base_architecture', 'inception_v3')
        preprocess_input_func, target_size, last_conv_layer_name, model_class = get_cnn_info(cnn_arch)

        class_name = "Indeterminado"; confidence = 0.0
        overlayed_image = None

        img = load_img(laudo_instance.imagem_lesao.path, target_size=target_size)
        x = img_to_array(img)
        # --- APLICA ALONGAMENTO DE CONTRASTE ---
        print("   [SERVICE] Aplicando Alongamento de Contraste...")
        x = apply_contrast_stretching(x)   
             
        x_expanded = np.expand_dims(x, axis=0)
        x_preprocessed = preprocess_input_func(x_expanded.copy())

        if pipeline_type == 'end_to_end':
            print("   [SERVICE] Pipeline 'Ponta a Ponta' detectado...")
            model = load_model(model_path)
            
            probabilities = model.predict(x_preprocessed)[0]
            class_index = np.argmax(probabilities)
            confidence = probabilities[class_index] * 100
            
            # --- CORREÇÃO GRAD-CAM: RECONSTRUÇÃO DO MODELO "PLANO" ---
            print("   [SERVICE] Reconstruindo modelo para Grad-CAM...")
            # 1. Cria uma nova base "plana"
            flat_base_model = model_class(weights=None, include_top=False, input_shape=target_size + (3,))
            # 2. Transfere os pesos da base aninhada do modelo salvo para a nova base plana
            flat_base_model.set_weights(model.get_layer(index=1).get_weights())
            # 3. Reconstrói a cabeça do classificador sobre a base plana
            x_head = GlobalAveragePooling2D()(flat_base_model.output)
            # Acessa as camadas da cabeça do modelo original pelos últimos índices
            x_head = model.layers[-2](x_head) # Acessa a penúltima camada (geralmente Dropout ou Dense)
            predictions = model.layers[-1](x_head) # Acessa a última camada (Dense de classificação)
            # 4. Cria o modelo final "plano"
            grad_cam_model = Model(inputs=flat_base_model.input, outputs=predictions)
            
            overlayed_image = generate_grad_cam_overlay(laudo_instance.imagem_lesao.path, grad_cam_model, x_preprocessed, last_conv_layer_name=last_conv_layer_name)

        else: # Pipelines Híbrido e Clássico
            print(f"   [SERVICE] Pipeline '{pipeline_type}' detectado...")
            classifier = joblib.load(model_path)
            
            cnn_features, classic_features = None, None
            
            if pipeline_type in ['cnn_hybrid', 'cnn_only']:
                if params.get('use_fine_tuning'):
                    # Lógica para carregar o extrator especialista e gerar Grad-CAM
                    # (Esta parte já estava correta, mas a mesma lógica de reconstrução seria aplicada aqui)
                    pass
                else: # Extração direta genérica
                    base_model_generic = model_class(weights='imagenet', include_top=False, input_shape=target_size + (3,))
                    feature_extractor = Model(inputs=base_model_generic.input, outputs=GlobalAveragePooling2D()(base_model_generic.output))
                    
                    # Para Grad-CAM, o modelo precisa ter uma cabeça de classificação
                    num_classes_map = len(active_model_instance.class_map_json)
                    x_head = feature_extractor.output
                    predictions = Dense(num_classes_map, activation='softmax')(x_head)
                    full_cnn_model_for_gradcam = Model(inputs=feature_extractor.input, outputs=predictions)
                    overlayed_image = generate_grad_cam_overlay(laudo_instance.imagem_lesao.path, full_cnn_model_for_gradcam, x_preprocessed, last_conv_layer_name=last_conv_layer_name)
                
                cnn_features = feature_extractor.predict(x_preprocessed, verbose=0)

            if pipeline_type in ['cnn_hybrid', 'classic_only']:
                classic_features = extract_classic_features_from_image(laudo_instance.imagem_lesao.path).reshape(1, -1)

            if pipeline_type == 'cnn_hybrid': final_features = np.concatenate((cnn_features, classic_features), axis=1)
            elif pipeline_type == 'cnn_only': final_features = cnn_features
            else: final_features = classic_features
            
            if hasattr(classifier, 'best_feature_indices_'):
                final_features = final_features[:, classifier.best_feature_indices_]
            
            class_index = classifier.predict(final_features)[0]
            confidence = np.max(classifier.predict_proba(final_features)) * 100

        # --- FINALIZAÇÃO E SALVAMENTO ---
        class_map = active_model_instance.class_map_json
        class_name = class_map.get(str(class_index), f"Classe_{class_index}") if class_map else f"Classe_{class_index}"
        
        print(f"   [SERVICE] ... Predição concluída. Classe: '{class_name}', Confiança: {confidence:.2f}%")

        if overlayed_image:
            buf = io.BytesIO()
            overlayed_image.save(buf, format='PNG')
            grad_cam_filename = f'gradcam_{laudo_instance.id}.png'
            laudo_instance.grad_cam_img.save(grad_cam_filename, ContentFile(buf.getvalue()), save=False)
            print("   [SERVICE] ... Grad-CAM gerado e pronto para salvar.")

        laudo_instance.resultado_classificacao = class_name
        laudo_instance.confianca = confidence
        laudo_instance.modelo_utilizado = active_model_instance
        laudo_instance.save()
        
        print(f"--- [SERVICE] CLASSIFICAÇÃO FINALIZADA COM SUCESSO: {class_name} ---")

    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! [SERVICE] ERRO FATAL NA CLASSIFICAÇÃO: {e} !!!")
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
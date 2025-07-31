# ml_manager/tasks.py (Refatorado com a Lógica de Pipeline Solicitada)

from celery import shared_task
import time, os, zipfile, shutil, joblib, io, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import unicodedata

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from tensorflow.keras.applications import InceptionV3, ResNet50, EfficientNetB0, DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model

from django.conf import settings
from django.core.files.base import ContentFile
from laudos.models import ModeloML
from .feature_extractor import extract_classic_features_from_image
from .genetic_algorithm import run_ga_feature_selection

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO, AUMENTO E UTILIDADE ---
def sanitize_filename(name):
    nfkd_form = unicodedata.normalize('NFKD', name)
    sanitized_name = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return sanitized_name.replace(' ', '_')

def apply_contrast_stretching(image_array):
    img_float = image_array.astype(float)
    for i in range(3):
        min_val, max_val = np.min(img_float[:,:,i]), np.max(img_float[:,:,i])
        if max_val > min_val:
            img_float[:,:,i] = 255 * (img_float[:,:,i] - min_val) / (max_val - min_val)
    return img_float.astype(np.uint8)

def add_gaussian_noise(image_array):
    mean = 0; std_dev = 0.1
    noise = np.random.normal(mean, std_dev, image_array.shape)
    return np.clip(image_array + noise, 0, 255)

def offline_augmentation(df_to_augment, angles=[15, 45]):
    new_rows = []
    print(f"   Iniciando aumento de dados offline para {len(df_to_augment)} imagens de treino...")
    for index, row in tqdm(df_to_augment.iterrows(), total=len(df_to_augment)):
        img_path = row['filename']
        img = load_img(img_path)
        img_array = img_to_array(img)
        base_path, original_filename = os.path.split(img_path)
        filename, ext = os.path.splitext(original_filename)
        augmentations = {
            f"{filename}_flip{ext}": np.fliplr(img_array),
            f"{filename}_noise{ext}": add_gaussian_noise(img_array),
            f"{filename}_rot15{ext}": tf.keras.preprocessing.image.apply_affine_transform(img_array, theta=15),
            f"{filename}_rot-15{ext}": tf.keras.preprocessing.image.apply_affine_transform(img_array, theta=-15),
            f"{filename}_rot45{ext}": tf.keras.preprocessing.image.apply_affine_transform(img_array, theta=45),
            f"{filename}_rot-45{ext}": tf.keras.preprocessing.image.apply_affine_transform(img_array, theta=-45),
        }
        for new_filename, new_img_array in augmentations.items():
            new_path = os.path.join(base_path, new_filename)
            save_img(new_path, new_img_array)
            new_rows.append({'filename': new_path, 'label_id': row['label_id'], 'label_name': row['label_name']})
    return pd.concat([df_to_augment, pd.DataFrame(new_rows)], ignore_index=True)

def get_image_paths_and_labels(base_path):
    image_paths, labels, class_names = [], [], sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if not class_names and os.listdir(base_path):
        sub_dir_name = os.listdir(base_path)[0]; potential_path = os.path.join(base_path, sub_dir_name)
        if os.path.isdir(potential_path): base_path = potential_path; class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if not class_names: raise ValueError("Nenhuma pasta de classe encontrada no dataset.")
    label_map = {name: i for i, name in enumerate(class_names)}
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        for f in os.listdir(class_path):
            if f.lower().endswith(valid_extensions): image_paths.append(os.path.join(class_path, f)); labels.append(label_map[class_name])
    return np.array(image_paths), np.array(labels), class_names, base_path

def generate_boxplot(data, title, metric_name, version_name):
    plt.figure(figsize=(8, 6)); sns.boxplot(data=data); plt.title(title); plt.ylabel(metric_name)
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    image_name = f'boxplot_{metric_name.lower().replace(" ", "_")}_{version_name}.png'; image_file = ContentFile(buf.read(), name=image_name)
    buf.close(); plt.close(); return image_name, image_file

def calculate_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specificities = []
    for i in range(n_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        specificities.append(specificity)
    return np.mean(specificities)

def generate_confusion_matrix(y_true_all, y_pred_all, class_names, version_name):
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'Matriz de Confusão Agregada para {version_name}'); plt.ylabel('Classe Verdadeira'); plt.xlabel('Classe Predita')
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    image_name = f'cm_{version_name}.png'; image_file = ContentFile(buf.read(), name=image_name)
    buf.close(); plt.close(); return image_name, image_file

def get_cnn_model(architecture_name):
    target_size = (224, 224) if architecture_name != 'inception_v3' else (299, 299)
    if architecture_name == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=target_size + (3,))
        return base_model, preprocess_input, target_size
    elif architecture_name == 'efficientnet_b0':
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=target_size + (3,))
        return base_model, preprocess_input, target_size
    elif architecture_name == 'densenet_201':
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=target_size + (3,))
        return base_model, preprocess_input, target_size
    else:
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=target_size + (3,))
        return base_model, preprocess_input, target_size

# --- TAREFA CELERY PRINCIPAL ---
@shared_task
def run_experimental_training_task(model_id, zip_path, options):
    model_instance = ModeloML.objects.get(id=model_id)
    try:
        overall_start_time = time.time()
        
        print(f"\n### ETAPA A: Preparando o dataset...")
        extract_path_base = os.path.join(settings.MEDIA_ROOT, 'datasets', model_instance.nome_versao)
        if os.path.exists(extract_path_base): shutil.rmtree(extract_path_base)
        with zipfile.ZipFile(zip_path, 'r') as zf: zf.extractall(extract_path_base)
        image_paths, y_all, class_names_final, _ = get_image_paths_and_labels(extract_path_base)
        
        print("\n### PRÉ-PROCESSAMENTO: Aplicando Alongamento de Contraste a todas as imagens...")
        for path in tqdm(image_paths, desc="Aplicando Contrast Stretching"):
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                stretched_img_rgb = apply_contrast_stretching(img_rgb)
                stretched_img_bgr = cv2.cvtColor(stretched_img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, stretched_img_bgr)
        print("   ...Alongamento de Contraste concluído.")

        num_classes = len(class_names_final)
        df_full = pd.DataFrame({'filename': image_paths, 'label_id': y_all, 'label_name': [class_names_final[i] for i in y_all]})

        pipeline_type = options.get('pipeline_type')
        cnn_arch = options.get('base_architecture', 'inception_v3')
        
        if pipeline_type == 'end_to_end':
            # --- PIPELINE "PONTA A PONTA" ---
            print("\n### INICIANDO PIPELINE 'PONTA A PONTA' (CNN COMO CLASSIFICADOR) ###")
            
            train_df, val_df = train_test_split(df_full, test_size=0.20, random_state=42, stratify=df_full['label_id'])
            
            if options.get('apply_augmentation'):
                train_df = offline_augmentation(train_df.copy())

            base_model, preprocess_input_func, target_size = get_cnn_model(cnn_arch)

            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_func)
            val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_func)
            train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filename', y_col='label_name', target_size=target_size, batch_size=32, class_mode='categorical', classes=class_names_final, shuffle=True)
            validation_generator = val_datagen.flow_from_dataframe(val_df, x_col='filename', y_col='label_name', target_size=target_size, batch_size=32, class_mode='categorical', classes=class_names_final, shuffle=False)
            
            base_model.trainable = False
            inputs = Input(shape=target_size + (3,)); x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x); x = Dropout(0.3)(x); predictions = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs, predictions)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            print("\n   --- Treinando a cabeça do classificador ---")
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[early_stopping], verbose=1)

            if options.get('use_fine_tuning'):
                print("\n   --- Realizando Ajuste Fino (Fine-Tuning) ---")
                base_model.trainable = True
                fine_tune_at = len(base_model.layers) // 2 
                for layer in base_model.layers[:fine_tune_at]: layer.trainable = False
                
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[early_stopping], verbose=1)

            print("\n   Avaliando o modelo final no conjunto de validação...")
            y_proba = model.predict(validation_generator)
            y_pred = np.argmax(y_proba, axis=1)
            y_true = validation_generator.classes
            
            model_filename = f"{model_instance.nome_versao}.h5"
            model.save(os.path.join(settings.MEDIA_ROOT, 'ml_models', model_filename))
            model_instance.arquivo_modelo.name = os.path.join('ml_models', model_filename)

            model_instance.acuracia_media = accuracy_score(y_true, y_pred)
            model_instance.sensibilidade_media = recall_score(y_true, y_pred, average='macro', zero_division=0)
            model_instance.especificidade_media = calculate_specificity(y_true, y_pred, num_classes)
            model_instance.kappa_media = cohen_kappa_score(y_true, y_pred)
            if len(np.unique(y_true)) > 1:
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                model_instance.auc_roc_media = roc_auc_score(y_true_bin, y_proba, multi_class='ovr')

            cm_name, cm_file = generate_confusion_matrix(y_true, y_pred, class_names_final, model_instance.nome_versao)
            model_instance.matriz_confusao_img.save(cm_name, cm_file, save=False)

        else: # cnn_hybrid, cnn_only, classic_only
            print(f"\n### INICIANDO PIPELINE '{pipeline_type}' ###")

           # if options.get('apply_augmentation'):
            #    X_full = offline_augmentation(train_df.copy())
            
            # --- ETAPA 1: EXTRAÇÃO DE CARACTERÍSTICAS ---
            print(f"\n### Etapa 1: Extraindo características...")
            cnn_features, classic_features = None, None
            _, preprocess_input_func, target_size = get_cnn_model(cnn_arch)

            if pipeline_type in ['cnn_hybrid', 'cnn_only']:
                print(f"   Usando extrator genérico ({cnn_arch}) da ImageNet")
                base_model_generic, _, _ = get_cnn_model(cnn_arch)
                feature_extractor = Model(inputs=base_model_generic.input, outputs=GlobalAveragePooling2D()(base_model_generic.output))
                all_cnn_features = [feature_extractor.predict(preprocess_input_func(np.expand_dims(img_to_array(load_img(p, target_size=target_size)), axis=0)), verbose=0).flatten() for p in tqdm(df_full['filename'].values, desc="Extraindo Features CNN")]
                cnn_features = np.array(all_cnn_features)

            if pipeline_type in ['cnn_hybrid', 'classic_only']:
                 print("   Extraindo features clássicas...")
                 classic_features_list = [extract_classic_features_from_image(p) for p in tqdm(df_full['filename'].values, desc="Extraindo Features Clássicas")]
                 classic_features = np.array(classic_features_list)

            if pipeline_type == 'cnn_hybrid': X_full = np.concatenate((cnn_features, classic_features), axis=1)
            elif pipeline_type == 'cnn_only': X_full = cnn_features
            else: X_full = classic_features
            y_full = df_full['label_id'].values
            print(f"   Extração concluída. Shape das características: {X_full.shape}")

            # --- ETAPA 2: SELEÇÃO DE CARACTERÍSTICAS (OPCIONAL) ---
            best_feature_indices = None
            if options.get('feature_selection_method') == 'genetic_algorithm':
                print("\n### Etapa 2: Selecionando características com Algoritmo Genético...")
                best_feature_indices = run_ga_feature_selection(X_full, y_full, X_full.shape[1])
                X_full = X_full[:, best_feature_indices]
                print(f"   Seleção concluída. Novo shape: {X_full.shape}")

            # --- ETAPA 3: DIVISÃO PRINCIPAL (80/20) ---
            print("\n### Etapa 3: Dividindo dados em Treino (80%) e Teste (20%)...")
            X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.20, random_state=42, stratify=y_full)

            # --- ETAPA 4: BALANCEAMENTO ÚNICO COM SMOTE (APENAS NO TREINO) ---
            print("\n### Etapa 4: Aplicando SMOTE ao conjunto de treino...")
            smote = SMOTE(k_neighbors=options.get('smote_k_neighbors', 3), random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            print(f"   Tamanho do treino antes do SMOTE: {X_train.shape}")
            print(f"   Tamanho do treino após o SMOTE: {X_train_smote.shape}")
            
            # --- ETAPA 5: VALIDAÇÃO CRUZADA REPETIDA (NO TREINO BALANCEADO) ---
            print("\n### Etapa 5: Executando Validação Cruzada Repetida...")
            k = options.get('k_folds', 10); n_repeats = options.get('n_repeats', 10)
            
            # Lógica de k-folds dinâmico e seguro
            _, train_smote_counts = np.unique(y_train_smote, return_counts=True)
            min_samples_cv = train_smote_counts.min()
            final_k = min(k, min_samples_cv)
            if final_k < k:
                print(f"   AVISO: O número de dobras (k={k}) é maior que o número de amostras na menor classe ({min_samples_cv}) do conjunto de treino balanceado.")
                print(f"   O valor de k foi ajustado para {final_k} para garantir a execução.")
            if final_k < 2:
                raise ValueError(f"Não é possível realizar a validação cruzada. A menor classe no conjunto de treino tem apenas {min_samples_cv} amostra(s).")
            
            validator = RepeatedStratifiedKFold(n_splits=final_k, n_repeats=n_repeats, random_state=42)
            fold_metrics = {'acuracia': [], 'sensibilidade': [], 'especificidade': [], 'auc_roc': [], 'kappa': []}
            
            classifier_name = options.get('final_classifier', 'random_forest')
            if classifier_name == 'random_forest': 
                classifier_cv = RandomForestClassifier(n_estimators=options.get('rf_n_estimators', 100), max_depth=options.get('rf_max_depth') or None, random_state=42, n_jobs=-1)
            elif classifier_name == 'svm': 
                classifier_cv = SVC(C=options.get('svm_c', 1.0), kernel=options.get('svm_kernel', 'rbf'), probability=True, random_state=42)
            else: 
                hidden_layers_str = options.get('mlp_hidden_layer_sizes', '100,'); hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip());
                classifier_cv = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=options.get('mlp_max_iter', 200), random_state=42, early_stopping=True)

            for fold, (train_idx, val_idx) in enumerate(validator.split(X_train_smote, y_train_smote)):
                if (fold + 1) % 10 == 0:
                    print(f"--- Processando Repetição/Fold {fold + 1}/{final_k * n_repeats} ---")
                X_t, X_v, y_t, y_v = X_train_smote[train_idx], X_train_smote[val_idx], y_train_smote[train_idx], y_train_smote[val_idx]
                classifier_cv.fit(X_t, y_t)
                y_pred, y_proba = classifier_cv.predict(X_v), classifier_cv.predict_proba(X_v)
                
                fold_metrics['acuracia'].append(accuracy_score(y_v, y_pred))
                fold_metrics['sensibilidade'].append(recall_score(y_v, y_pred, average='macro', zero_division=0))
                fold_metrics['especificidade'].append(calculate_specificity(y_v, y_pred, num_classes))
                fold_metrics['kappa'].append(cohen_kappa_score(y_v, y_pred))
                if len(np.unique(y_v)) > 1:
                    y_val_bin = label_binarize(y_v, classes=range(num_classes))
                    if num_classes > 2: fold_metrics['auc_roc'].append(roc_auc_score(y_val_bin, y_proba, multi_class='ovr'))
                    else: fold_metrics['auc_roc'].append(roc_auc_score(y_v, y_proba[:, 1]))
            
            model_instance.acuracia_media = np.mean(fold_metrics['acuracia']) if fold_metrics['acuracia'] else 0; model_instance.acuracia_std = np.std(fold_metrics['acuracia']) if fold_metrics['acuracia'] else 0
            model_instance.sensibilidade_media = np.mean(fold_metrics['sensibilidade']) if fold_metrics['sensibilidade'] else 0; model_instance.sensibilidade_std = np.std(fold_metrics['sensibilidade']) if fold_metrics['sensibilidade'] else 0
            model_instance.especificidade_media = np.mean(fold_metrics['especificidade']) if fold_metrics['especificidade'] else 0; model_instance.especificidade_std = np.std(fold_metrics['especificidade']) if fold_metrics['especificidade'] else 0
            model_instance.auc_roc_media = np.mean(fold_metrics['auc_roc']) if fold_metrics['auc_roc'] else 0; model_instance.auc_roc_std = np.std(fold_metrics['auc_roc']) if fold_metrics['auc_roc'] else 0
            model_instance.kappa_media = np.mean(fold_metrics['kappa']) if fold_metrics['kappa'] else 0; model_instance.kappa_std = np.std(fold_metrics['kappa']) if fold_metrics['kappa'] else 0
            model_instance.metricas_raw = fold_metrics
            
            # --- ETAPA 6: TREINO E TESTE FINAL ---
            print("\n### Etapa 6: Treinando modelo final e avaliando no conjunto de teste...")
            final_classifier = classifier_cv
            final_classifier.fit(X_train_smote, y_train_smote)
            
            y_pred_final = final_classifier.predict(X_test)
            
            cm_name, cm_file = generate_confusion_matrix(y_test, y_pred_final, class_names_final, model_instance.nome_versao)
            model_instance.matriz_confusao_img.save(cm_name, cm_file, save=False)
            
            metric_to_field_map = {'acuracia': ('boxplot_acuracia_img', 'Acurácia'), 'sensibilidade': ('boxplot_sensibilidade_img', 'Sensibilidade'), 'especificidade': ('boxplot_especificidade_img', 'Especificidade'), 'auc_roc': ('boxplot_auc_roc_img', 'AUC-ROC'), 'kappa': ('boxplot_kappa_img', 'Kappa')}
            for metric_name, (field_name, plot_title) in metric_to_field_map.items():
                if fold_metrics.get(metric_name):
                    title = f'Box Plot - {plot_title}'; boxplot_name, boxplot_file = generate_boxplot(fold_metrics[metric_name], title, plot_title, model_instance.nome_versao)
                    getattr(model_instance, field_name).save(boxplot_name, boxplot_file, save=False)
            
            if options.get('final_classifier') == 'random_forest' and hasattr(final_classifier, 'feature_importances_'):
                print("   ... Calculando e salvando a importância das características...")
                feature_names = []
                if pipeline_type in ['cnn_hybrid', 'cnn_only']:
                    num_cnn_features = cnn_features.shape[1]
                    feature_names.extend([f"CNN_{i}" for i in range(num_cnn_features)])
                if pipeline_type in ['cnn_hybrid', 'classic_only']:
                    classic_feature_names = ['area','perimeter','eccentricity','solidity','contrast','dissimilarity','homogeneity','energy_glcm','correlation','energy_wavelet_cA','energy_wavelet_cH','energy_wavelet_cV','energy_wavelet_cD','shannon_entropy']
                    feature_names.extend(classic_feature_names)
                
                final_feature_names = [feature_names[i] for i in best_feature_indices] if best_feature_indices is not None else feature_names
                importances = final_classifier.feature_importances_
                indices = np.argsort(importances)[-20:]
                
                plt.figure(figsize=(12, 10)); plt.title('Top 20 Features Mais Importantes')
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [final_feature_names[i] for i in indices]); plt.xlabel('Importância Relativa'); plt.tight_layout()
                
                buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
                fi_image_name = f'fi_{model_instance.nome_versao}.png'; fi_image_file = ContentFile(buf.read(), name=fi_image_name)
                plt.close(); model_instance.importancia_features_img.save(fi_image_name, fi_image_file, save=False)
                
                top_features_data = {final_feature_names[i]: float(importances[i]) for i in indices}
                model_instance.top_features_json = top_features_data
            
            if best_feature_indices is not None:
                final_classifier.best_feature_indices_ = best_feature_indices
            model_filename = f"{model_instance.nome_versao}.joblib"
            joblib.dump(final_classifier, os.path.join(settings.MEDIA_ROOT, 'ml_models', model_filename))
            model_instance.arquivo_modelo.name = os.path.join('ml_models', model_filename)

        # --- FINALIZAÇÃO ---
        model_instance.parametros_treinamento = options
        model_instance.tempo_treinamento_seg = time.time() - overall_start_time
        class_map = {i: name for i, name in enumerate(class_names_final)}; model_instance.class_map_json = class_map
        model_instance.save()
        print("--- PROCESSO TOTALMENTE CONCLUÍDO ---")

    except Exception as e:
        print(f"!!! ERRO NO TREINAMENTO: {e} !!!"); traceback.print_exc()
        model_instance.cenario_clinico += f" (FALHOU: {e})"; model_instance.save()
    finally:
        if os.path.exists(zip_path):
            try: os.remove(zip_path)
            except OSError as e: print(f"Erro ao remover arquivo zip temporário: {e}")
        print("--- Limpeza de arquivos temporários concluída. ---")
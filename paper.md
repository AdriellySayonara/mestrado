---
title: 'DermAI: An Open-Source Web Framework for Hybrid Deep Learning in the Differential Diagnosis of Leprosy and Skin Cancer'
tags:
  - Python
  - Django
  - deep learning
  - medical imaging
  - skin lesion classification
  - transfer learning
  - explainable AI
authors:
  - name: Fernando Antonio de Oliveira
    orcid: 0009-0002-9865-2834
    affiliation: "1"
  - name: Adrielly Sayonara de Oliveira Silva
    orcid: 0009-0004-2220-8053
    affiliation: "1"
  - name: José Barbosa de Araújo Neto
    orcid: 0009-0007-4994-1695
    affiliation: "1"
  - name: Camila Tiodista de Lima
    orcid: 0009-0007-6013-7059
    affiliation: "1"
  - name: Flávio Secco Fonseca
    orcid: 0000-0003-4956-1135
    affiliation: "2"
  - name: Ana Clara Gomes da Silva
    orcid: 0000-0002-2823-5763
    affiliation: "2"
  - name: Clarisse Lima
    orcid: 0000-0003-1198-8627
    affiliation: "1"
  - name: Maíra Araújo de Santana
    orcid: 0000-0002-1796-7862
    affiliation: "1"
  - name: Juliana Carneiro Gomes
    orcid: 0000-0002-0785-0767
    affiliation: "1"
  - name: Giselle Machado Magalhães Moreno 
    orcid: 0000-0003-4076-3494
    affiliation: "1"
  - name: Wellington Pinheiro dos Santos
    orcid: 0000-0003-2558-6602
    affiliation: "1 , 2"
# affiliations:
  - name: Department of Biomedical Engineering, Federal University of Pernambuco, Brazil
    index: 1
  - name: Graduate Program in Computer Engineering, University of Pernambuco, Brazil
    index: 2
date: 5 April 2025
bibliography: paper.bib
---

# Summary

`DermAI` is an open-source, web-based research software for the differential diagnosis of leprosy and skin cancer using clinical and dermoscopic images. Built with Django and leveraging a hybrid machine learning pipeline, it combines pre-trained convolutional neural networks (CNNs)—EfficientNetB0, InceptionV3, and DenseNet201—as feature extractors with classical classifiers (MLP, Random Forest, SVM) [@fonseca2022early], [@tawhid2021spectrogram] for robust lesion classification. The system supports six clinically motivated diagnostic scenarios, ranging from multi-class differentiation of leprosy subtypes to binary discrimination between malignant and benign lesions.

A core innovation is the integrated **Model Factory**, which enables researchers to experiment with different combinations of feature extractors, data balancing (SMOTE), lesion segmentation (U-Net), and explainable AI (Grad-CAM). The software also provides a clinician-facing interface that generates interpretable diagnostic reports with visual heatmaps highlighting regions of interest.

`DermAI` addresses a critical gap in digital dermatology for neglected tropical diseases, particularly in low-resource settings where leprosy remains endemic [@barbieri2022reimagining]. By providing a modular, reusable, and transparent platform, it supports both clinical decision support and reproducible research in medical image analysis[@tan2025clinical].

# Statement of Need

Current open-source tools for skin lesion analysis rarely address the **differential diagnosis between infectious and neoplastic skin conditions**, especially neglected diseases like leprosy [@oliveira2025dermai]. Moreover, many existing systems lack modularity, interpretability, or integration into a deployable clinical workflow.

`DermAI` fills this niche by offering:
- A **hybrid modeling approach** that decouples feature extraction from classification, enabling flexible experimentation;
- Support for **multi-scenario evaluation**, reflecting real-world diagnostic challenges;
- Built-in **explainability** via Grad-CAM to foster clinician trust;
- A **full-stack web application** following the MVC pattern (Django), making it deployable in clinical or research environments.

The software has been validated on the public *Atlas Dermatológico* dataset [@atlasdermatologico] and demonstrates >94% accuracy in distinguishing leprosy from skin cancer and >98% accuracy in separating skin cancer from benign mimickers.

# Features

- **Modular pipeline**: Swap CNN backbones and classifiers without code changes.
- **Data preprocessing**: Includes U-Net-based lesion segmentation and classical feature extraction (GLCM, wavelets).
- **Class imbalance handling**: SMOTE and image augmentation.
- **Model evaluation**: Stratified k-fold cross-validation with metrics (accuracy, sensitivity, specificity, AUC-ROC, Cohen’s Kappa).
- **Explainable AI**: Grad-CAM heatmaps embedded in diagnostic reports.
- **Role-based interface**: Separate views for patients, physicians, and ML engineers.

# Quality Control

`DermAI` is implemented in Python 3 using standard scientific libraries (TensorFlow/Keras, scikit-learn) and the Django web framework. The repository includes:
- A `requirements.txt` for dependency management;
- Structured Django app architecture with clear separation of concerns;
- Example training scripts and model serialization;
- Automated tests (under development) and continuous integration (planned).

The software is distributed under the **GNU General Public License v3.0**, ensuring broad academic and commercial reuse.

# Acknowledgements

This work was developed as part of a Master’s dissertation at the Federal University of Pernambuco (UFPE), Brazil, with support from the Graduate Program in Biomedical Engineering. The authors thank the *Atlas Dermatológico* team for providing public access to dermatological images.

# References
# DermAI: An Open-Source Web Framework for Hybrid Deep Learning in the Differential Diagnosis of Leprosy and Skin Cancer

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg )](https://www.gnu.org/licenses/gpl-3.0 )


`DermAI` is an open-source research software for non-invasive, image-based differential diagnosis of **leprosy**, **skin cancer**, and other dermatological lesions. Built with Django and leveraging a **hybrid machine learning pipeline**, it combines pre-trained convolutional neural networks (CNNs)—EfficientNetB0, InceptionV3, DenseNet201—as feature extractors with classical classifiers (MLP, Random Forest, SVM) for robust lesion classification.

The system supports **six clinically motivated diagnostic scenarios**, ranging from multi-class differentiation of leprosy subtypes to binary discrimination between malignant and benign lesions. A core innovation is the integrated **Model Factory**, which enables researchers to experiment with different combinations of feature extractors, data balancing (SMOTE), lesion segmentation (U-Net), and explainable AI (Grad-CAM).

**Disclaimer**: This is a **Minimum Viable Product (MVP) for research purposes only**. It is **not a medical device** and **must not be used for clinical diagnosis without professional supervision**.

---

##  Key Features

- **6 diagnostic scenarios** simulating real-world clinical challenges.
- **Modular hybrid pipeline**: decoupled CNN feature extraction + classical ML classification.
- **Lesion segmentation** using U-Net (ResNet34 backbone).
- **Class imbalance handling** via SMOTE in feature space and image augmentation.
- **Explainable AI (XAI)** with Grad-CAM heatmaps embedded in diagnostic reports.
- **Role-based web interface** for:
- **Physicians**: upload images → receive interpretable diagnostic reports.
- **Patients**: view medical reports.
- **ML Engineers**: train, compare, and analyze models via the "Model Factory".

---

##  Installation

### Prerequisites
- Python ≥ 3.8
- pip
- Git

## Steps


### 1. Clone the repository
```
git clone https://github.com/AdriellySayonara/mestrado.git
cd mestrado
```

### 2. Create and activate a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

```
### 3. Install dependencies
```
pip install -r requirements.txt
```

**Note:** requirements.txt includes TensorFlow, scikit-learn, Django, Pillow, segmentation-models, and other scientific libraries.

## Local Execution
DermAI is a Django web application. To run it locally:

### 1. Apply database migrations
```
python manage.py migrate
```

### 2. (Optional) Create a superuser for admin access
```
python manage.py createsuperuser
```

### 3. Start the development server
```
python manage.py runserver
```

Access the application at: http://127.0.0.1:8000


- Use the superuser account to access the admin panel (/admin) and activate models for production.
- The physician interface allows image upload and generates diagnostic reports with Grad-CAM heatmaps.

# Testing

The repository includes basic integration tests. To run them:

```
python manage.py test

```

**Planned expansion:** Unit tests for model loading, image preprocessing, and Grad-CAM generation. Contributions are welcome!

# Documentation

Comprehensive documentation is available in the Repository Wiki , including:

- Tutorial for physicians (interpreting diagnostic reports)
- ML Engineer’s guide to the "Model Factory"
- Description of the 6 clinical scenarios
- Explanation of the hybrid pipeline architecture
- Instructions for training new models

# License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) — see the LICENSE file for details.

# Citation

```
@phdthesis{oliveira2025dermai,
  title={Sistema inteligente para apoio ao diagnóstico da hanseníase e outras lesões baseado em imagens de lesões de pele},
  author={Oliveira, Fernando Antonio de},
  year={2025},
  school={Universidade Federal de Pernambuco}
}
```
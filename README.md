
# Intelligent System for Supporting the Diagnosis of Skin Lesions

This project is a web platform developed in Django, a Minimum Viable Product or Proof of Concept, designed as a tool for identifying leprosy and other skin lesions through lesion images. Within the project, modules were also created for the training, evaluation, and comparative analysis of Machine Learning pipelines for the classification of skin lesion images. The system was developed as part of a master’s dissertation and implements advanced techniques, including hybrid feature extraction, optimization with evolutionary algorithms, and data augmentation.

## Main Features

- **User Interfaces:** Dedicated dashboards for Doctors and Patients.  
- **Model Factory:** A research interface to configure and train multiple AI models with customizable parameters.  
- **Advanced Pipelines:** Support for classical and Deep Learning feature extraction, SMOTE, cGANs, and Genetic Algorithms.  
- **Results Analysis:** A consolidated dashboard to compare the performance of different experiments using statistical metrics and box plot charts.  
- **Explainable AI (XAI):** Generation of feature importance plots and Grad-CAM heatmaps.  
- **PDF Reports:** Automatic generation of patient reports.  

## Prerequisites

Before getting started, make sure you have the following software installed on your system:

- **Python:** Version 3.11.x  
- **Git:** To clone the repository.  
- *On Windows, you can install it from the [official Microsoft release](https://github.com/microsoftarchive/redis/releases)*  
- **GTK+ for Windows (Windows only):** This is an **external** system dependency required for the `WeasyPrint` library (PDF generation) to work.  
  - 1. Download the installer at: [GTK+ for Windows Runtime Environment](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).  
  - 2. Run the installer and follow the steps.  

## Installation Guide

Follow the steps below to set up the development environment and run the project locally.

### Clone the Repository

Open your terminal and clone the GitHub repository:
```
git clone [YOUR_REPOSITORY_URL_HERE]
cd [PROJECT_FOLDER_NAME]
```
# Create the virtual environment
```
python -m venv venv
```
# Activate the virtual environment
## On Windows:
```
venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```
### Configure the Database
```
python manage.py migrate
```
### Create a Superuser (Administrator)
```
python manage.py createsuperuser
```
## **Step 4: How to Run the Project**

## Running the Project

To run the full system, you will need **two separate terminals**, both with the virtual environment activated.

### Terminal 1: Start the Django Server

This terminal will run the web application.

```
python manage.py runserver
```

### Terminal 2: Start the Celery Worker

### On Windows, use the '-P solo' flag for stability
celery -A config worker --loglevel=info -P solo
Important: Make sure your Redis service is running in the background before starting the Celery worker.


## **Step 5: How to Use the System**


## How to Use

1.  **Access the Admin:** Go to `http://127.0.0.1:8000/admin/` and log in with your superuser account.
2.  **Register a Doctor and a Patient:**
    -   First, create `Users` for a doctor and a patient.
    -   Then, go to the "Doctors" and "Patients" sections to create the corresponding profiles, linking the patient to the doctor.
3.  **Sign Up as a Doctor:** Alternatively, go to the home page and use the "Sign Up as Doctor" option.
4.  **Run an Experiment:**
    -   Log in as superuser.
    -   Go to the **"Model Factory"** menu.
    -   Configure an experiment, upload a dataset (`.zip` with folders per class), and start the training. Track the progress in the Celery terminal.
5.  **Use the System as a Doctor:**
    -   Log in with the doctor account.
    -   On your dashboard, register new patients or request the analysis of an image for an existing patient.
    -   Check the history and click on a report to see the details, including the XAI analysis (Grad-CAM) and the option to download the PDF.
6. **Analyze Results:**
    -   Go to the **"Results Dashboard"** menu.
    -   Use the filters to select a scenario and a classifier and view the comparative charts.

## Project Structure

-   `config/`: Main Django configurations (`settings.py`, `urls.py`).
-   `core/`: App for central pages and logic (home page).
-   `users/`: App for user management (Doctors, Patients).
-   `laudos/`: App for report logic, real-time classification, and visualizations.
-   `ml_manager/`: App for the "Model Factory" and the "Results Dashboard".

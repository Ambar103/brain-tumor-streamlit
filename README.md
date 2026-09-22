# Brain Tumor MRI Classifier

## Overview
This project is a brain tumor MRI classifier built using Streamlit, a powerful framework for creating web applications in Python. It leverages deep learning techniques to analyze MRI images and predict the presence of brain tumors.

LINK TO TRY IT OUT : https://brain-tumor-app-cbtaz6mjq8x6lc9mquzkcj.streamlit.app/

## Model
An EfficientNetB0 backbone was fine-tuned on the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (glioma, meningioma, pituitary, no tumor — ~7,000 MRI images). Training used transfer learning in two phases (frozen head, then fine-tuning the top layers) with data augmentation. Test-set accuracy: **82.5%**. The trained model is hosted on [Hugging Face Hub](https://huggingface.co/Ambar10/brain-tumor-efficientnetb0) and downloaded automatically by the app. See `training/train.py` for the full pipeline and `training/artifacts/` for the confusion matrix and classification report.

The app also shows a **Grad-CAM** heatmap overlay for every prediction, highlighting which region of the MRI most influenced the model's decision.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Ambar103/brain-tumor-streamlit.git
   ```
2. Navigate to the project directory:
   ```bash
   cd brain-tumor-streamlit
   ```
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```bash
streamlit run app.py
```
Open your web browser and go to `http://localhost:8501` to view the application.

## Dependencies
This project requires the following Python packages:
- streamlit
- tensorflow
- huggingface_hub
- numpy
- Pillow
- matplotlib

You can find a complete list of dependencies in the `requirements.txt` file.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.



---

## Acknowledgments
- Thanks to [Streamlit](https://streamlit.io/) for providing an easy way to create web apps.
- Thanks to all contributors and the open-source community for their support and guidance in building this project.

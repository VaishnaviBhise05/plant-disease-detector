# Plant Disease Detection (CNN + Streamlit)

Upload a leaf photo and a CNN predicts the plant
disease. Trained on the PlantVillage dataset (38 classes).
<img width="1133" height="910" alt="Screenshot 2026-11-03 101322" src="https://github.com/user-attachments/assets/76e958a7-17d8-484c-b5c4-e656ce4762f6" />


## Project structure

```
plant-disease-detector/
├── train.py             # trains the model, saves plant_model.keras + class_names.json
├── app.py               # Streamlit web app (upload leaf -> prediction)
├── requirements.txt     # Python libraries
├── PlantVillage/        # PUT THE DATASET HERE
├── .vscode/             # VS Code settings + Run/Debug buttons
└── README.md
```

## Step 1: Download the dataset

1. Go to Kaggle and search **"PlantVillage dataset"** (about 54,000 images, 38 classes).
   A free Kaggle account is needed.
2. Download and unzip it.
3. Move the class folders into the `PlantVillage/` folder of this project.

## Step 2: Train the model

```
python train.py
```
Or use the **Run and Debug** panel (Ctrl+Shift+D) -> "Train model" -> green play.

- Takes about 10-30 minutes on a normal laptop CPU for the full dataset.
- At the end it prints validation accuracy and creates
  `plant_model.keras` and `class_names.json`.

## Step 3: Run the web app

```
streamlit run app.py
```
Your browser opens at http://localhost:8501. Upload a leaf photo (.jpg/.png)
and see the prediction, confidence and top 3 results.

<img width="1917" height="1013" alt="Screenshot 2026-11-03 101243" src="https://github.com/user-attachments/assets/d073d6e8-80c7-4279-bd8a-50407cbb270b" />


## Project report ideas

- Explain transfer learning and why MobileNetV2 (lightweight, fast)
- Show the dataset class distribution
- Add accuracy/loss graphs and a confusion matrix
- Discuss limitations (lab-style images vs real field photos)
- Future work: treatment advice in Marathi/Hindi, mobile app, field-photo dataset
  

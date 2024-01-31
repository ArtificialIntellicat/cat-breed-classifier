# Cat Breed Classifier

This project is a machine learning application that classifies images of cats into their respective breeds. It's built using TensorFlow and Flask and is designed as a portfolio project to demonstrate skills in machine learning, image processing, and web application development.

## Project Overview

The application allows users to upload a cat image and receive a prediction of the cat's breed. It includes a web interface for easy interaction. The backend uses a Convolutional Neural Network (CNN) model, trained on a large dataset of cat images, to perform the classification.

## Features

- Image upload functionality for breed prediction.
- A CNN model leveraging TensorFlow and Keras for image classification.
- A Flask web application for easy interaction with the model.
- Image validation to ensure corrupt images are handled appropriately.

## Installation

To set up and run this project locally, follow these steps:

1. **Clone the Repository:**
```
git clone https://github.com/your-github-username/cat-breed-classifier.git
cd cat-breed-classifier
```

2. **Create and Activate a Virtual Environment (optional but recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install Dependencies:**
```
pip install -r requirements.txt
```

4. **Train Model or Download Pre-Trained Model**

Either a) download a [Cat Breeds Dataset](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset/data), save it to a directory called 'data' in the root directory and train the model using main.py:
```
python main.py
```

Or b) download pre-trained model [here](https://drive.google.com/file/d/1eLju6NWcqlho4iLFX7pM5fjszsnmkD9V/view?usp=sharing) and save it to the directory 'models'.

4. **Run the Flask Application:**
```
python app.py
```

5. **Open a web browser and navigate to http://localhost:5000 to interact with the application.**

## Usage

After starting the Flask server, you can upload a cat image using the web interface. The model will process the image and display the predicted breed.

## Project Structure

- `app.py`: The Flask application file with routes and view logic.
- `model.py`: Contains the CNN model definitions.
- `main.py`: Script for training the model and handling image preprocessing.
- `requirements.txt`: List of Python dependencies for the project.
- `templates/`: Folder containing HTML templates for the web interface.
- `static/`: Folder for static files like CSS.
- `models/`: Directory where the trained model is saved.
- `data/`: Directory for storing cat images and other data used by the model.

## Contributing

Contributions to improve the application or extend its features are welcome. Please follow the standard fork-and-pull request workflow.

## License

This project is open-sourced under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to the open-source community and all contributors to the libraries and tools used in this project.

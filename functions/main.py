# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_admin import initialize_app

# initialize_app()
#
#
# @https_fn.on_request()
# def on_request_example(req: https_fn.Request) -> https_fn.Response:
#     return https_fn.Response("Hello world!")

from firebase_functions import https_fn
from fastai.vision.all import *
import json
import tempfile

# Load the model
learn = load_learner('model.pkl')

@https_fn.on_request()
def predict(req: https_fn.Request) -> https_fn.Response:
    if req.method == 'POST':
        file = req.files.get('file')
        if file:
            # Save the uploaded image temporarily
            with tempfile.NamedTemporaryFile() as temp_file:
                file.save(temp_file.name)
                img = PILImage.create(temp_file.name)

            # Get prediction from the model
            pred, idx, probs = learn.predict(img)
            result = {'Dog': float(probs[0]), 'Cat': float(probs[1])}

            # Return the prediction as JSON response
            return https_fn.Response(json.dumps(result), content_type='application/json')
        else:
            return https_fn.Response("No file uploaded", status=400)
    else:
        return https_fn.Response("Method not allowed", status=405)



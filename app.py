from flask import Flask, render_template, request, jsonify 
import pickle
import numpy as np 

# initialize the flask app 
app = Flask(__name__)

# load model 
model = pickle.load(open("model.pkl", "rb")) 

# classes - labels 
classes = ['Setosa', 'Versicolor', 'Virginica'] 

# Home route 
@app.route("/") 
def home(): 
    return render_template("index.html") 

# prediction route - web form 
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get the form values 
        features = [float(x) for x in request.form.values()]
        # convert to numpy array 
        final_input = np.array([features])

        #predict 
        prediction = model.predict(final_input) 

        
        output = classes[prediction[0]] 

        return render_template(
            "index.html",
            prediction_text=f"Predicted Flower: {output}"
        )
        
    except Exception as e: 
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        ) 
# REST API Route 
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try: 
        # getting data in json 
        data = request.get_json(force=True) 
        features = data["features"] 
        prediction = model.predict([features])  
        output = classes[prediction[0]] 

        return jsonify({
            "success": True,
            "prediction": output
        }) 

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }) 

#run flask app 
if __name__ == "__main__":  
    app.run(debug=True) 
    
        
        

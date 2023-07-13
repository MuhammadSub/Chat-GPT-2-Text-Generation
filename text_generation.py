from flask import Flask, request, render_template_string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer if they are not already loaded
if 'model' not in globals():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

# Process the Data 
def gen_text(input_text, tokenizer, model, max_length=200, early_stopping=True):
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')

    # Generate attention mask
    attention_mask = torch.ones_like(encoded_input)

    # Generate text
    output = model.generate(encoded_input.to(device), attention_mask=attention_mask.to(device), max_length=max_length, early_stopping=early_stopping)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded_output

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Process the POST request and generate the text
        user_input = request.form['user_input']
        max_length = int(request.form['max_length'])

        # Generate text
        generated_text = gen_text(user_input, tokenizer, model, max_length)

        return render_template_string('''
            <html>
            <head>
                <title>Text Generation</title>
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                <style>
                    .text-block {
                        background-color: #f2f2f2;
                        border: 1px solid #ccc;
                        padding: 10px;
                        margin-bottom: 20px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Text Generation</h1>
                    <form action="/" method="post">
                        <div class="form-group">
                            <label for="user_input">Enter your input:</label>
                            <input type="text" class="form-control" id="user_input" name="user_input">
                        </div>
                        <div class="form-group">
                            <label for="max_length">Enter the maximum length of the generated text:</label>
                            <input type="number" class="form-control" id="max_length" name="max_length">
                        </div>
                        <button type="submit" class="btn btn-primary">Generate Text</button>
                    </form>
                    {% if generated_text %}
                    <div class="text-block">
                        <h2>Generated Text:</h2>
                        <p><strong>Input:</strong></p>
                        <p>{{ generated_input }}</p>
                        <p><strong>Max Length:</strong></p>
                        <p>{{ generated_max_length }}</p>
                        <p><strong>Output:</strong></p>
                        <p>{{ generated_text }}</p>
                    </div>
                    {% endif %}
                </div>
                <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
                <script>
                    function clearFields() {
                        document.getElementById("user_input").value = "";
                    }
                </script>
            </body>
            </html>
            ''', user_input=user_input, generated_input=user_input, generated_max_length=max_length, generated_text=generated_text)
    else:
        return render_template_string('''
            <html>
            <head>
                <title>Text Generation</title>
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            </head>
            <body>
                <div class="container">
                    <h1>Text Generation</h1>
                    <form action="/" method="post">
                        <div class="form-group">
                            <label for="user_input">Enter your input:</label>
                            <input type="text" class="form-control" id="user_input" name="user_input">
                        </div>
                        <div class="form-group">
                            <label for="max_length">Enter the maximum length of the generated text:</label>
                            <input type="number" class="form-control" id="max_length" name="max_length">
                        </div>
                        <button type="submit" class="btn btn-primary">Generate Text</button>
                    </form>
                </div>
                <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
                <script>
                    function clearFields() {
                        document.getElementById("user_input").value = "";
                    }
                </script>
            </body>
            </html>
            ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

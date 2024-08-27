from flask import Flask, render_template, request
from rag_pipeline import process_rag_query

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = process_rag_query(question)
        return render_template('result.html', question=question, answer=answer)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
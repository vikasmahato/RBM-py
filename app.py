from flask import Flask
import rbm
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

app = Flask(__name__)


@app.route('/getPredictions', methods=('GET', 'POST'))
def getPredictions():
    print(request)
    userid = request.form['userid']
    return rbm.predict(int(userid))


@app.route('/', methods=('GET', 'POST'))
def login():

    return render_template('index.html')


if __name__ == '__main__':
    rbm.rbm()
    app.run()

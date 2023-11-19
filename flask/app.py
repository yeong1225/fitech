from flask import Flask, render_template, Response
from views.main_views import bp 
from views.detect_views import dbp



app = Flask(__name__)

# Blueprint를 /search 경로에 등록
app.register_blueprint(bp, url_prefix='/')
app.register_blueprint(dbp, url_prefix='/')




@app.route('/search')
def hello_world():
    return 'Hello /search route'

if __name__ == '__main__':
    app.run()







from flask import Flask, Blueprint

bp = Blueprint('baker_proj', __name__, template_folder='/Users/amarkanaka/repos/pageStreamSegmentation/frontend/my-svelte-app/src/templates', static_folder="/Users/amarkanaka/repos/pageStreamSegmentation/frontend/my-svelte-app/src/static")

@bp.route("/")
def initial():
    return "Hello!"

app = Flask(__name__)
# register blueprint here for prefix
app.register_blueprint(bp, url_prefix='/')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
import os

from flask import Flask, request, jsonify
from flask import send_from_directory

app = Flask(__name__)


# 在 Flask 中有一个路由的概念，我们将路由可以理解为外部通过什么样的路径可以触发我们的这个函数，
# 在这个小例子中，通过根路径就可以访问到我们的 hello_world 函数，因此，这里 app.route() 装饰器传入的就是“/”。
@app.route('/')
def hello_world():
    return 'Hello World!'


# 设置图标
@app.route('/favicon.ico')
def favicon():
    # D:\PycharmProjects\xiebo\diantou\sina
    # print(app.root_path)
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/hello_rec', methods=["POST"])
def hello_recommendation():
    # noinspection PyBroadException
    try:
        if request.method == 'POST':
            req_json = request.get_json()
            user_id = req_json["user_id"]
            # Serialize the given arguments as JSON, and return a :class:`~flask.Response` object with the ``application/json`` mimetype.
            # A dict or list returned from a view will be converted to a JSON response automatically without needing to call this.
            return jsonify({"code": 200, "msg": "请求成功", "data": "hello " + str(user_id)})
    except Exception as e:
        return jsonify({"code": 2000, "msg": str(e)})


if __name__ == '__main__':
    #  * Serving Flask app 'app'
    #  * Debug mode: off
    # WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    #  * Running on http://127.0.0.1:5000
    app.run(host='127.0.0.1', port=5000, debug=False)

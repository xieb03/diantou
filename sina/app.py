import hashlib
import os

from flask import Flask, request, jsonify
from flask import send_from_directory

from sina.dao.mysql_db import Mysql
from sina.entity.user import User
from util.page_utils import PageUtils

page_utils = PageUtils()

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
    try:
        if request.method == 'POST':
            req_json = request.get_json()
            user_id = req_json["user_id"]
            # Serialize the given arguments as JSON, and return a :class:`~flask.Response` object with the ``application/json`` mimetype.
            # A dict or list returned from a view will be converted to a JSON response automatically without needing to call this.
            return jsonify({"code": 200, "msg": "请求成功", "data": "hello " + str(user_id)})
    except Exception as e:
        return jsonify({"code": 2000, "msg": str(e)})


@app.route("/recommendation/get_rec_list", methods=['POST'])
def get_rec_list():
    if request.method == 'POST':
        req_json = request.get_json()
        page_num = req_json['page_num']
        page_size = req_json['page_size']
        try:
            data = page_utils.get_data_with_page(page_num, page_size)
            return jsonify({"code": 0, "msg": "请求成功", "data": data})
        except Exception as e:
            return jsonify({"code": 2000, "msg": str(e)})


@app.route("/recommendation/register", methods=['POST'])
def register():
    if request.method == 'POST':
        req_json = request.get_json()
        user = User()
        user.username = req_json['username']
        user.nick = req_json['nick']
        user.age = req_json['age']
        user.gender = req_json['gender']
        user.city = req_json['city']
        # 密码使用 MD5 的方式进行加密
        user.password = str(hashlib.md5(req_json['password'].encode()).hexdigest())

        try:
            mysql = Mysql()
            with mysql.db_session() as sess:
                if sess.query(User.id).filter(User.username == user.username).count() > 0:
                    return jsonify({"code": 1000, "msg": "用户已存在"})
                sess.add(user)
                sess.commit()
            result = jsonify({"code": 0, "msg": "注册成功"})
            return result
        except Exception as e:
            return jsonify({"code": 2000, "msg": str(e)})


@app.route("/recommendation/login", methods=['POST'])
def login():
    if request.method == 'POST':
        req_json = request.get_json()
        username = req_json['username']
        # 由于 MD5 是不可逆的加密方式，因此校验方法是将用户输入的密码再一次进行 MD5 加密，然后再和数据库中的密码进行比对
        password = str(hashlib.md5(req_json['password'].encode()).hexdigest())

        try:
            mysql = Mysql()
            with mysql.db_session() as sess:
                # 这里需要注意的是，在登录时返回了一个 userid，这样以后再发其他请求的时候，就可以用 userid 来作为唯一的标识了。
                res = sess.query(User.id).filter(User.username == username, User.password == password)
            if res.count() > 0:
                for x in res.all():
                    data = {"userid": str(x[0])}
                    info = jsonify({"code": 0, "msg": "登录成功", "data": data})
                    return info
            else:
                return jsonify({"code": 1000, "msg": "用户名或密码错误"})
        except Exception as e:
            return jsonify({"code": 2000, "msg": str(e)})


if __name__ == '__main__':
    #  * Serving Flask app 'app'
    #  * Debug mode: off
    # WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    #  * Running on http://127.0.0.1:5000
    app.run(host='127.0.0.1', port=5000, debug=False)

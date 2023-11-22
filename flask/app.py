from flask import Flask, request, redirect, url_for, render_template, session, flash
from flask_mysqldb import MySQL
#from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 보안을 위한 시크릿 키 설정

# Configure MySQL connection
app.config['MYSQL_HOST'] = 'database-3.cdha8hslaur6.ap-northeast-2.rds.amazonaws.com'  # Replace with your MySQL host
app.config['MYSQL_USER'] = 'root'  # Replace with your MySQL username
app.config['MYSQL_PASSWORD'] = 'root1234'  # Replace with your MySQL password
app.config['MYSQL_DB'] = 'users'  # Replace with your MySQL database name

mysql = MySQL(app)




# 홈페이지
@app.route('/')
def home():
        return render_template('home.html')


# 기록 페이지
@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

# 연락처 페이지
@app.route('/contact')
def contact():
    return render_template('contact.html')

# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        user_password = request.form['user_password']

        cursor = mysql.connection.cursor()
       # user_name도 검색하도록 쿼리 수정
        cursor.execute("SELECT user_name, user_password FROM users2 WHERE user_id = %s", (user_id,))
        user_record = cursor.fetchone()
        cursor.close()

        if user_record and user_record[1] == user_password: # 비밀번호가 user_record[1]에 위치#check_hash 권장
            # 비밀번호가 일치하는 경우
            session['user_id'] = user_id
            session['user_name'] = user_record[0]  # user_name을 세션에 저장
            return redirect(url_for('home'))
        else:
            # 비밀번호가 일치하지 않는 경우
            flash("Email이나 Password가 일치하지 않습니다.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

# 회원가입 페이지
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 폼 데이터에서 정보 검색
        user_name = request.form['user_name']
        user_id = request.form['user_id']
        user_password = request.form['user_password']
        

        # MySQL에 연결
        cursor = mysql.connection.cursor()

        # user_id가 이미 존재하는지 확인
        cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
        if cursor.fetchone():
            flash("ID가 이미 존재합니다. 다른 ID를 선택해주세요.", 'error')
            return redirect(url_for('register'))

# 비밀번호 해시 처리
        #hashed_password = generate_password_hash(user_password)

        # 데이터베이스에 사용자 정보와 해시된 비밀번호 저장
        #cursor.execute("INSERT INTO users (user_name, user_id, user_password) VALUES (%s, %s, %s)", (user_name, user_id, hashed_password))

        # 새 사용자 데이터 삽입
        cursor.execute("INSERT INTO users2 (user_name, user_id, user_password) VALUES (%s, %s, %s)", (user_name, user_id, user_password))
        mysql.connection.commit()
        cursor.close()

        # 사용자가 등록되었으므로 세션 설정/ 사용자 로그인
        session['user_id'] = user_id
        session['user_name']=user_name

        # 성공 페이지 또는 홈페이지로 리디렉션
        return redirect(url_for('home'))

    return render_template('register.html')
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # 세션에서 user_id 제거
    session.pop('user_name', None) 
    return redirect(url_for('home'))




if __name__ == '__main__':
    app.run(debug=True)

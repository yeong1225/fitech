from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify

cal = Blueprint('cal', __name__, template_folder="templates")
from db_config import db  # db_config 모듈에서 db 불러오기

from flask import jsonify
import datetime


@cal.route('/calendar')
def calendar():
    return render_template('calendar.html')


@cal.route('/add_cal', methods=['POST'])
def add_cal():
    user_id = request.form.get('user_id')
    date = request.form.get('date')
    time = int(request.form.get('time'))
    memo = request.form.get('memo')

    cursor = db.cursor()
    query = "INSERT INTO calendar4 (user_id, date, time, memo) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (user_id, date, time, memo))
    db.commit()
    cursor.close()
    return redirect(url_for('ctest'))

@cal.route('/get_event', methods=['GET', 'POST'])
def get_event():
    # 현재 로그인한 사용자의 user_id 가져오기
    user_id = session.get('user_id')

    # 선택한 날짜를 request.args에서 가져오거나 다른 방법으로 입력합니다.
    selected_date = request.args.get('selected_date')

    # 날짜에 해당하는 데이터를 DB에서 검색합니다.
    cursor = db.cursor()
    query = "SELECT time, memo FROM calendar4 WHERE user_id = %s AND date = %s"
    cursor.execute(query, (user_id, selected_date,))
    data = cursor.fetchall()
    cursor.close()
    
    #print('검색 결과:', data)
    #print('반환되는 데이터:', events)

    # 데이터를 JSON 형식으로 변환하여 반환합니다.
    events = [{'time': row[0], 'memo': row[1]} for row in data]

    return jsonify(events)



    

    
# 운동 후 페이지
@cal.route('/add_memo', methods=['POST'])
def add_memo():
    if not session.get('user_id'):
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    try:
        # 요청으로부터 데이터 추출
        data = request.get_json()
        user_id = session.get('user_id')
        date = data.get('date')
        memo = data.get('memo')

        # 문자열을 날짜 객체로 변환
        date_obj = datetime.datetime.now().strftime('%Y-%m-%d')

        # 데이터베이스에 쿼리 실행
        cursor = db.cursor()
        query = "UPDATE calendar4 SET memo = %s WHERE user_id = %s AND date = %s"
        cursor.execute(query, (memo, user_id, date_obj))
        db.commit()
        cursor.close()
        #cursor.execute("SELECT * FROM users2 WHERE user_id = %s", (user_id,))
        return jsonify({'status': 'success'})
    except Exception as e:
        # 예외 발생 시 오류 메시지 반환
        return jsonify({'status': 'error', 'message': str(e)}), 500

    
# 캘린더 페이지
@cal.route('/after')
def after():
    
    time_difference = session.get('time_difference', 'No data')
    
    #user_id = session.get('user_id')
    user_id = session.get('user_id')
    print(user_id)
    if user_id:
        cursor = db.cursor()
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        query = """
        INSERT INTO calendar4 (user_id, date, time)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE time = %s
        """
        cursor.execute(query, (user_id, current_date, time_difference, time_difference))
        db.commit()
         # 쿼리 실행 결과 로깅
        if cursor.rowcount > 0:
            print(f"Success: {cursor.rowcount} row(s) affected.")
        else:
            print("No rows affected.")
        cursor.close()




    # Pass the grade, color, hits, and total to your template
    return render_template('after.html', time_difference=time_difference)


@cal.route('/send_selected_value', methods=['POST'])
def send_selected_value():
    try:
        user_id = session.get('user_id')
        if user_id:
            cursor = db.cursor()
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            selected_value = request.form.get('selectedValue')  # 선택한 값을 가져옴

            query = """
            INSERT INTO button2 (user_id, date, state)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE state = %s
            """
            cursor.execute(query, (user_id, current_date, selected_value, selected_value))
            db.commit()

            cursor.close()

        return "Selected value sent successfully"
    except Exception as e:
        return jsonify({'error': str(e)})


@cal.route('/get_audio_state', methods=['POST'])
def get_audio_state():
    try:
         user_id = session.get('user_id')
         if user_id:
             
            user_id = session.get('user_id')  # 사용자 ID 가져오기
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')  # 현재 날짜 가져오기

            # 데이터베이스에서 user_id와 date에 해당하는 state 값을 가져옴
            cursor = db.cursor()
            query = "SELECT state FROM button2 WHERE user_id = %s AND date = %s"
            cursor.execute(query, (user_id, current_date))
            result = cursor.fetchone()
            cursor.close()

            state = result[0] if result else None  # state 값을 가져와서 state 변수에 저장

            return jsonify(state)

        
    except Exception as e:
        return jsonify({'error': str(e)})


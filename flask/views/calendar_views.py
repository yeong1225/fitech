from flask import Blueprint, render_template, request, redirect, url_for, session, flash

cal = Blueprint('cal', __name__, template_folder="templates")
from db_config import db  # db_config 모듈에서 db 불러오기

from flask import jsonify
import datetime


# 이벤트 추가(캘린더)
@cal.route('/add_event', methods=['POST'])
def add_event():
    if not session.get('user_id'):
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    try:
        # 요청으로부터 데이터 추출
        data = request.get_json()
        user_id = session.get('user_id')
        date = data.get('date')
        memo = data.get('memo')

        # 문자열을 날짜 객체로 변환
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()

        # 데이터베이스에 쿼리 실행
        cursor = db.cursor()
        query = "INSERT INTO calendar2 (user_id, date, memo) VALUES (%s, %s, %s)"
        cursor.execute(query, (user_id, date_obj, memo))
        db.commit()
        cursor.close()

        return jsonify({'status': 'success'})
    except Exception as e:
        # 예외 발생 시 오류 메시지 반환
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 기록 페이지
@cal.route('/calendar')
def calendar_page():
    user_id = session['user_id']
    try:
        cursor = db.cursor()
        query = "SELECT id, date, time, list, memo FROM calendar2 WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        events = cursor.fetchall()
        cursor.close()

        # Format events for JavaScript
        formatted_events = [{
            'id': event[0],
            'date': event[1].strftime('%Y-%m-%d'),
            'time': event[2],  # 'time' 값을 추가합니다.
            'list': event[3],  # 'list' 값을 추가합니다.
            'memo': event[4]
        } for event in events]

        return render_template('calendar.html', events=formatted_events)
    except Exception as e:
        print(e)
        return render_template('calendar.html', events=[])

@cal.route('/delete_event', methods=['POST'])
def delete_event():
    if not session.get('user_id'):
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    try:
        data = request.get_json()
        date = data.get('date')
        memo = data.get('memo')


        cursor = db.cursor()
        query = "DELETE FROM calendar2 WHERE date = %s AND memo = %s"
        cursor.execute(query, (date, memo))
        db.commit()
        cursor.close()

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

    
# 이벤트추가
@cal.route('/add_memo', methods=['POST'])
def add_memo():
    if not session.get('user_id'):
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    try:
        # 요청으로부터 데이터 추출
        data = request.get_json()
        user_id = session.get('user_id')
        time = data.get('time')
        list = ', '.join(data['list'])  # 배열을 문자열로 변환
        date = data.get('date')
        memo = data.get('memo')

        # 문자열을 날짜 객체로 변환
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()

        # 데이터베이스에 쿼리 실행
        cursor = db.cursor()
        query = "INSERT INTO calendar2 (user_id, date, time, list, memo) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (user_id, date_obj, time, list, memo))
        db.commit()
        cursor.close()

        return jsonify({'status': 'success'})
    except Exception as e:
        # 예외 발생 시 오류 메시지 반환
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
# 운동 후 페이지
@cal.route('/after')
def after():
    hits = 10  # Replace with actual logic to get this value
    total = 14  # Replace with actual logic to get this value

    def calculate_grade(hits, total):
        if total == 0:
            return 'N/A', '#555555'  # Dark gray for undefined grade

        percentage = (hits / total) * 100

        if percentage >= 90:
            return 'A', '#4c9a2a'  # Dark green
        elif percentage >= 80:
            return 'B', '#2a77ad'  # Dark blue
        elif percentage >= 70:
            return 'C', '#ebcb2d'  # Dark yellow
        elif percentage >= 60:
            return 'D', '#ad6c2a'  # Dark orange
        else:
            return 'F', '#9a2a2a'  # Dark red

    grade, color = calculate_grade(hits, total)

    # Pass the grade, color, hits, and total to your template
    return render_template('after.html', grade=grade, color=color, hits=hits, total=total)
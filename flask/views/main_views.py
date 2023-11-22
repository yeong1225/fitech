from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from scipy import spatial
import math
from flask import jsonify
from flask_socketio import SocketIO, emit

# 모듈 import
#import pymysql


bp = Blueprint('main', __name__, template_folder="templates")


mp_drawing = mp.solutions.drawing_utils # Visualizing our poses
mp_pose = mp.solutions.pose # Importing our pose estimation model (ex)hand,


@bp.route('/')
def home():
    return render_template('home.html')

@bp.route('/calendar')
def calendar():
    return render_template('calendar.html')

@bp.route('/login')
def login():
    return render_template('login.html')


@bp.route('/register')
def register():
    return render_template('register.html')


@bp.route('/contact')
def contact():
    return render_template('contact.html')





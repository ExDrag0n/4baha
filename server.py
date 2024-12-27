import time
from flask import Flask, request, send_from_directory, abort, jsonify, render_template
import os
import cv2
import numpy as np
from main import start_process
import ipaddress
from threading import Thread, Event
import uuid
from io import BytesIO
from pathlib import Path
import asyncio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './output_images'

MAX_CONCURRENT_REQUESTS = 3
current_concurrent_requests = 0

# Глобальная очередь задач и флаг для остановки обработки
task_queue = []
stop_processing = Event()

def clear_all(filename_male):
    if len(os.listdir('4pres')) != 0:
        for i in range(1, 8):
            if os.path.isfile(f'4pres/{filename_male}_{i}.png'):
                    os.remove(f'4pres/{filename_male}_{i}.png')
                    print(f'4pres/{filename_male}_{i}.png deleted')
    if len(os.listdir('output_images')) != 0:
        if os.path.isfile(f'output_images/{filename_male}.jpg'):
            os.remove(f'output_images/{filename_male}.jpg')
            print(f'output_images/{filename_male}.jpg cleared')
    if len(os.listdir('output_images')) != 0:
        for i in range(7):
            if os.path.isfile(f'output_images/{filename_male}_000{i}.png'):
                os.remove(f'output_images/{filename_male}_000{i}.png')
                print(f'output_images/{filename_male}_000{i}.png cleared')
    if len(os.listdir('results/males_model/test_latest/traversal')) != 0:
        for file in os.listdir('results/males_model/test_latest/traversal'):
            if os.path.isfile(f'results/males_model/test_latest/traversal/{filename_male}.mp4'):
                os.remove(f"results/males_model/test_latest/traversal/{filename_male}.mp4")
                print(f'age video (male) cleared {filename_male}.mp4')
    if len(os.listdir('results/females_model/test_latest/traversal')) != 0:
        for file in os.listdir('results/females_model/test_latest/traversal'):
            if os.path.isfile(f'results/females_model/test_latest/traversal/{filename_male}.mp4'):
                os.remove(f"results/females_model/test_latest/traversal/{filename_male}.mp4")
                print(f'age video (female) cleared {filename_male}.mp4')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


def process_task(task):
    global current_concurrent_requests
    current_concurrent_requests += 1

    gender = task['gender']
    race = task['race']
    filename = task['filename']

    filename = filename.split('.')[0]
    result = start_process(gender, race, filename)
    return result


def process_tasks():
    global current_concurrent_requests
    while not stop_processing.is_set():
        if task_queue:
            task = task_queue.pop(0)
            try:
                process_task(task)
                print(f"Обработка задачи {task} завершена")
            except Exception as e:
                print(f"Ошибка при обработке задачи {task}: {e}")
            current_concurrent_requests -= 1
            if current_concurrent_requests < MAX_CONCURRENT_REQUESTS and not threading.active_count() >= MAX_CONCURRENT_REQUESTS + 1:
                new_thread = Thread(target=process_tasks)
                new_thread.start()
        stop_processing.wait(0.1)


@app.route('/process', methods=['POST'])
async def process_image():
    if len(os.listdir('res_video')) != 0:
        for file in os.listdir('res_video'):
            os.remove(f"res_video/{file}")
    image_file = request.files['image']
    gender = request.form['gender']
    race = request.form['race']
    filename = str(uuid.uuid4())
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.jpg')
    image_file.save(image_path)

    if not image_file or not gender:
        return jsonify({'error': 'Отсутствуют обязательные поля'}), 400

    task = {
        'gender': gender,
        'race': race,
        'filename': f"{filename}"
    }

    task_queue.append(task)
    print(f"Текущее количество задач в очереди - {len(task_queue) + 1}")

    while not f"{filename}.mp4" in os.listdir('res_video'):
        await asyncio.sleep(5)
    result = f"{filename}.mp4"
    print(f'Отправка {result}')
    return send_from_directory("res_video", result), clear_all(filename)


if __name__ == '__main__':
    import threading

    processing_thread = threading.Thread(target=process_tasks)
    processing_thread.start()

    try:
        app.run(host="127.0.0.1", port=8080, debug=True)
    finally:
        stop_processing.set()

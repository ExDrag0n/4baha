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


async def process_task(task):
    global current_concurrent_requests
    current_concurrent_requests += 1

    gender = task['gender']
    race = task['race']
    filename = task['filename']

    filename = filename.split('.')[0]
    result = start_process(gender, race, filename)
    await result


async def process_tasks():
    global current_concurrent_requests
    while not stop_processing.is_set():
        if task_queue:
            task = task_queue.pop(0)
            try:
                await process_task(task)
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
##################################################################################################################################
##################################################################################################################################
##  БЫЛ ЕЩЁ ВАРИАНТ С ТРЕДАМИ, КОТОРЫЙ ДО ЭТОГО СЕРВ КРАШИЛ, НО Я ОПТИМИЗИРОВАЛ ПРОЦЕСС, ТЕПЕРЬ ХОТЬ 10 ПОТОКОВ ОДНОВРЕМЕННО    ##
##  НО ТАМ НА КОЛ-ВО ПОТОКОВ ОГРАНИЧЕНИЯ НЕТ + КОГДА ЗАПРОСОВ МНОГО НЕКОТОРЫЕ В БЕСКОНЧЕНОЕ ОЖИДАНИЕ УХОДЯТ ПО КАКОЙ ТО ПРИЧИНЕ ##
##  ЕГО КОД ПРИЛАГАЮ НИЖЕ. ЕЩЁ РАЗ СПАСИБО ЗА ПОМОЩЬ <3                                                                         ##
##################################################################################################################################
##################################################################################################################################
from flask import Flask, request, send_file, abort, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from main import start_process
import ipaddress
from threading import Thread
import threading
import uuid


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './output_images'


def clear_all(filename):
    if len(os.listdir('res_video')) != 0:
        for file in os.listdir('res_video'):
            for _ in os.listdir('res_video'):
                if os.path.isfile(f'res_video/{filename}.mp4'):
                    os.remove(f'res_video/{filename}.mp4')
        print(f'res_video cleared')
    if len(os.listdir('4pres')) != 0:
        for _ in os.listdir('4pres'):
            for i in range(1,8):
                if os.path.isfile(f'4pres/{filename}_{i}.png'):
                    os.remove(f'4pres/{filename}_{i}.png')
        print(f'4pres cleared')
    if len(os.listdir('output_images')) != 0:
        for _ in os.listdir('output_images'):
            for i in range(7):
                if os.path.isfile(f'output_images/{filename}_000{i}.png'):
                    os.remove(f'output_images/{filename}_000{i}.png')
        print(f'output_images cleared')
    if len(os.listdir('results/males_model/test_latest/traversal')) != 0:
        for file in os.listdir('results/males_model/test_latest/traversal'):
            for _ in os.listdir('results/males_model/test_latest/traversal'):
                if os.path.isfile(f'results/males_model/test_latest/traversal/{filename}.mp4'):
                    os.remove(f"results/males_model/test_latest/traversal/{filename}.mp4")
                    print(f'age video (male) cleared {filename}')
    if len(os.listdir('results/females_model/test_latest/traversal')) != 0:
        for file in os.listdir('results/females_model/test_latest/traversal'):
            if os.path.isfile(f'results/females_model/test_latest/traversal/{filename}.mp4'):
                os.remove(f"results/females_model/test_latest/traversal/{filename}.mp4")
                print(f'age video (female) cleared {filename}')

@app.route('/process', methods=['POST'])
def process_image():
    with open("dlt_lst.txt", "r") as f:
        dlt_file = f.readline()
    # client_ip = request.remote_addr
    # if client_ip not in ALLOWED_IPS:
    #     abort(403)

    image_file = request.files['image']
    gender = request.form['gender']
    race = request.form['race']

    if not image_file or not gender:
        return jsonify({'error': 'Отсутствуют обязательные поля'}), 400
    # Сохранение изображения
    filename = str(uuid.uuid4())
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.jpg')
    print(image_path)
    image_file.save(image_path)
    print('Запуск потока')
    thread = Thread(target=lambda: start_process(gender, race, filename))
    thread.start()
    thread.join()
    # каждый поток выполняет код ниже только по его завершении
    with open("dlt_lst.txt", "w") as delete_list:
        delete_list.write(filename)
    result = f"res_video/{filename}.mp4"
    return send_file(result, mimetype='video/mp4'), clear_all(filename)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
#########################################################################################################
###################                     ВАРИАНТ С FASTAPI                ################################
#########################################################################################################
import asyncio
from fastapi import FastAPI, BackgroundTasks, File, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
import os
import uuid
from queue import Queue
from contextlib import asynccontextmanager
from main import start_process

MAX_CONCURRENT_REQUESTS = 3
current_concurrent_requests = 0
task_queue = Queue()

ALLOWED_IPS = ["128.140.77.233"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Начинаем фоновую задачу для обработки очереди
    asyncio.create_task(process_tasks())
    yield


async def process_task(task):
    global current_concurrent_requests
    current_concurrent_requests += 1

    gender = task['gender']
    race = task['race']
    filename = task['filename']

    try:
        filename = filename.split('.')[0]
        result = await start_process(gender, race, filename)
        await result
    finally:
        current_concurrent_requests -= 1


async def process_tasks():
    global current_concurrent_requests
    while True:
        if current_concurrent_requests < MAX_CONCURRENT_REQUESTS and not task_queue.empty():
            task = task_queue.get_nowait()
            await process_task(task)
        else:
            # Если количество параллельных задач достигло лимита, ждём
            await asyncio.sleep(1)


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

app = FastAPI(lifespan=lifespan)


@app.post("/process")
async def process_image(background_tasks: BackgroundTasks, image: bytes = File(...), gender: str = Form(...), race: str = Form(...)):
    print(gender, race)
    buffer = BytesIO(image)
    filename = str(uuid.uuid4())
    image_path = os.path.join("./output_images", f"{filename}.jpg")
    with open(image_path, "wb") as f:
        f.write(buffer.getvalue())

    task = {
        'gender': gender,
        'race': race,
        'filename': f"{filename}"
    }

    task_queue.put_nowait(task)
    print(f"Текущее количество задач в очереди - {task_queue.qsize()}")

    # Очистка данных после обработки
    background_tasks.add_task(clear_all, filename)

    # Ожидание, пока видео не будет готово
    while not f"{filename}.mp4" in os.listdir('res_video'):
        print("Pending...")
        await asyncio.sleep(5)

    return StreamingResponse(generate_response(filename), media_type="video/mp4")


async def generate_response(filename):
    with open(f"res_video/{filename}.mp4", "rb") as video_file:
        while True:
            chunk = video_file.read(4096)
            if not chunk:
                break
            yield chunk


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)






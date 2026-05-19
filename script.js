const MODEL_PATH = './inf/cmcm.onnx';
const MODEL_SIZE = 640; // Стандартный размер входа YOLOv8
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

let session = null;
const uploadInput = document.getElementById('uploadInput');
const outputCanvas = document.getElementById('outputCanvas');
const ctx = outputCanvas.getContext('2d');
const statusText = document.getElementById('status');
const uploadBtn = document.querySelector('.upload-btn');

// Отключаем кнопку до загрузки модели
uploadBtn.classList.add('disabled');

// Инициализация ONNX Runtime сессии
async function loadModel() {
    try {
        // ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        // Пытаемся использовать WebGPU, фоллбэк на WASM
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['webgpu', 'wasm']
        });
        
        statusText.innerText = 'Модель готова к работе';
        statusText.style.color = '#4caf50';
        uploadBtn.classList.remove('disabled');
    } catch (error) {
        console.error("Ошибка загрузки модели:", error);
        statusText.innerText = 'Ошибка загрузки модели. Проверьте консоль.';
        statusText.style.color = '#f44336';
    }
}

// Обработка загрузки изображения
uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file || !session) return;

    statusText.innerText = 'Обработка...';
    
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        // 1. Отрисовка оригинала для сохранения пропорций
        outputCanvas.width = img.width;
        outputCanvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);

        // 2. Препроцессинг
        const tensor = preprocessImage(img);

        // 3. Инференс
        const feeds = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);
        
        // Сырой тензор: shape обычно [1, 84, 8400] для 80 классов COCO
        const output = results[session.outputNames[0]].data;

        // 4. Постпроцессинг (NMS)
        const boxes = postprocess(output, img.width, img.height);

        // 5. Отрисовка
        drawBoxes(boxes);
        
        statusText.innerText = `Найдено объектов: ${boxes.length}`;
        URL.revokeObjectURL(img.src);
    };
});

/**
 * ПРЕПРОЦЕССИНГ
 * Подготовка изображения к формату [1, 3, 640, 640] NCHW
 */
function preprocessImage(img) {
    // Используем скрытый canvas для ресайза до 640x640
    const offCanvas = document.createElement('canvas');
    offCanvas.width = MODEL_SIZE;
    offCanvas.height = MODEL_SIZE;
    const offCtx = offCanvas.getContext('2d');
    
    // Рисуем изображение, масштабируя его (в идеале нужно сохранять aspect ratio с паддингом, 
    // но для простоты YOLO часто переваривает прямое сжатие)
    offCtx.drawImage(img, 0, 0, MODEL_SIZE, MODEL_SIZE);
    
    // Извлекаем пиксели: массив [R, G, B, A, R, G, B, A...]
    const imgData = offCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;

    // В памяти создаем непрерывный блок для R, G, B (формат Planar/NCHW)
    const red = new Float32Array(MODEL_SIZE * MODEL_SIZE);
    const green = new Float32Array(MODEL_SIZE * MODEL_SIZE);
    const blue = new Float32Array(MODEL_SIZE * MODEL_SIZE);

    for (let i = 0; i < imgData.length; i += 4) {
        const pixelIndex = i / 4;
        // Нормализация 0-255 -> 0.0-1.0
        red[pixelIndex] = imgData[i] / 255.0;
        green[pixelIndex] = imgData[i + 1] / 255.0;
        blue[pixelIndex] = imgData[i + 2] / 255.0;
    }

    // Собираем в единый одномерный массив (Tensor Data)
    const tensorData = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
    tensorData.set(red, 0);
    tensorData.set(green, MODEL_SIZE * MODEL_SIZE);
    tensorData.set(blue, 2 * MODEL_SIZE * MODEL_SIZE);

    return new ort.Tensor('float32', tensorData, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

/**
 * ПОСТПРОЦЕССИНГ
 * YOLOv8 отдает транспонированный тензор: [1, 4 + num_classes, 8400]
 * В плоском (flattened) виде память организована так: сначала 8400 значений cx, 
 * затем 8400 значений cy, затем w, h, и затем 8400 вероятностей для каждого класса.
 */
function postprocess(output, origWidth, origHeight) {
    const numAnchors = 8400; // Количество grid cells / предсказаний (YOLOv8)
    const numClasses = (output.length / numAnchors) - 4;
    const boxes = [];

    // Масштабные коэффициенты для возврата координат к исходному размеру картинки
    const scaleX = origWidth / MODEL_SIZE;
    const scaleY = origHeight / MODEL_SIZE;

    for (let i = 0; i < numAnchors; i++) {
        let maxClassScore = 0;
        let classId = -1;

        // Поиск класса с максимальной уверенностью для данного анкора
        for (let c = 0; c < numClasses; c++) {
            // Индекс в плоском массиве: (Смещение фичи * количество_анкоров) + текущий_анкор
            const score = output[(4 + c) * numAnchors + i];
            if (score > maxClassScore) {
                maxClassScore = score;
                classId = c;
            }
        }

        // Фильтрация по порогу (Confidence Threshold)
        if (maxClassScore >= CONFIDENCE_THRESHOLD) {
            const cx = output[0 * numAnchors + i];
            const cy = output[1 * numAnchors + i];
            const w = output[2 * numAnchors + i];
            const h = output[3 * numAnchors + i];

            // Конвертация [center_x, center_y, width, height] -> [x1, y1, x2, y2]
            // и сразу масштабируем под размер оригинального изображения
            const x1 = (cx - w / 2) * scaleX;
            const y1 = (cy - h / 2) * scaleY;
            const x2 = (cx + w / 2) * scaleX;
            const y2 = (cy + h / 2) * scaleY;

            boxes.push({ x1, y1, x2, y2, score: maxClassScore, classId });
        }
    }

    // Применяем Non-Maximum Suppression (NMS) для удаления дубликатов
    return nms(boxes, IOU_THRESHOLD);
}

/**
 * Алгоритм Non-Maximum Suppression (Жадный подход)
 */
function nms(boxes, iouThreshold) {
    // Сортируем рамки по убыванию уверенности (score)
    boxes.sort((a, b) => b.score - a.score);
    const selectedBoxes = [];

    while (boxes.length > 0) {
        // Берем рамку с наибольшим score
        const currentBox = boxes.shift();
        selectedBoxes.push(currentBox);

        // Оставляем только те рамки, которые слабо пересекаются с текущей (IoU < порог)
        // или относятся к другому классу
        for (let i = boxes.length - 1; i >= 0; i--) {
            const box = boxes[i];
            if (box.classId === currentBox.classId) {
                const iou = calculateIoU(currentBox, box);
                if (iou > iouThreshold) {
                    boxes.splice(i, 1);
                }
            }
        }
    }
    return selectedBoxes;
}

// Расчет Intersection over Union
function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
}

/**
 * ОТРИСОВКА (Draw)
 */
function drawBoxes(boxes) {
    ctx.lineWidth = Math.max(2, outputCanvas.width / 300); // Динамическая толщина линии
    ctx.font = `${Math.max(16, outputCanvas.width / 40)}px sans-serif`;
    ctx.textBaseline = "top";

    boxes.forEach(box => {
        // Цвет (можно сделать массив цветов для разных классов)
        ctx.strokeStyle = "#bb86fc"; 
        ctx.fillStyle = "#bb86fc";

        // Рисуем Bounding Box
        ctx.beginPath();
        ctx.rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        ctx.stroke();

        // Текст с классом и вероятностью
        const text = `Class ${box.classId} (${Math.round(box.score * 100)}%)`;
        const textWidth = ctx.measureText(text).width;
        const textHeight = parseInt(ctx.font, 10);

        // Подложка для текста
        ctx.fillRect(box.x1, box.y1 - textHeight - 4, textWidth + 8, textHeight + 4);
        
        ctx.fillStyle = "#000000";
        ctx.fillText(text, box.x1 + 4, box.y1 - textHeight - 2);
    });
}

// Запуск инициализации при загрузке скрипта
loadModel();

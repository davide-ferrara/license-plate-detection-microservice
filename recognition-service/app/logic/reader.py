# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
This implementation does its best to follow the Robert Martin's Clean code guidelines.
The comments follows the Google Python Style Guide:
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2023, FCRlab at University of Messina'
__author__ = 'Davide Ferrara <frrdvd98m07h224l@studenti.umime.it>, Lorenzo Carnevale <lcarnevale@unime.it>'
__credits__ = ''
__description__ = 'Reader class'

import os
import cv2
import time
import torch
import logging
import threading
import numpy as np
import easyocr
import csv
import torch.backends.cudnn as cudnn
from PIL import Image
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.params import Parameters

class Reader:

    def __init__(self, static_files_potential, static_files_detection, shared, model_path, mutex, verbosity, logging_path) -> None:
        self.__static_files_potential = static_files_potential
        self.__static_files_detection = static_files_detection
        self.__shared = shared
        self.__mutex = mutex
        self.__reader = None
        self.__params = Parameters(model_path)
        self.__model, self.__labels = self.__load_yolov5_model()
        self.__text_reader = self.__easyocr_model_load()
        self.__setup_logging(verbosity, logging_path)

    def __setup_logging(self, verbosity, path):
        format = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s"
        filename = path
        datefmt = "%d/%m/%Y %H:%M:%S"
        level = logging.INFO
        if verbosity:
            level = logging.DEBUG
        logging.basicConfig(filename=filename, filemode='a', format=format, level=level, datefmt=datefmt)

    def setup(self):
        if not os.path.exists(self.__static_files_detection):
            os.makedirs(self.__static_files_detection)

        self.__reader = threading.Thread(
            target=self.__reader_job,
            args=()
        )

    def __reader_job(self):
        while True:
            if not self.__potential_folder_is_empty():
                self.__mutex.acquire()
                oldest_frame_path = self.__oldest()

                frame = self.__get_frame(oldest_frame_path)

                detected, cropped = self.__detection(frame, self.__model, self.__labels)
                os.remove(oldest_frame_path)

                self.__mutex.release()

                detected_image = Image.fromarray(detected)
                filename = os.path.basename(oldest_frame_path)
                absolute_path = f'{self.__static_files_detection}/{filename}'
                detected_image.save(absolute_path)

                if cropped is not None:
                    plate_num = self.__easyocr_model_works(cropped)
                    filtered_plate_num = self.__filter_plate_numbers(plate_num)

                    logging.info(f"Detected plate: {filtered_plate_num}")

                    logging.info(f"Saving results into csv file...")
                    self.__save_results(filtered_plate_num, os.path.join(self.__shared, 'results.csv'), self.__mutex)

                time.sleep(0.1)

    def __potential_folder_is_empty(self):
        path = self.__static_files_potential
        return True if not len(os.listdir(path)) else False

    def __oldest(self):
        path = self.__static_files_potential
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        return min(paths, key=os.path.getctime)

    def __load_yolov5_model(self):
        """
        It loads the model and returns the model and the names of the classes.
        :return: model, names
        """
        model = attempt_load(self.__params.model, map_location=self.__params.device)
        print("device", self.__params.device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        return model, names

    def __get_frame(self, filename):
        """ Read image from file using opencv.

            Args:
                filename(str): relative or absolute path of the image

            Returns:
                (numpy.ndarray) frame read from file 
        """
        return cv2.imread(filename)

    def __detection(self, frame, model, names):
        detected_plate = frame.copy()
        cropped_plate = None
        frame = cv2.resize(frame, (self.__params.pred_shape[1], self.__params.pred_shape[0]), interpolation=cv2.INTER_LINEAR)
        frame = np.transpose(frame, (2, 1, 0))
        cudnn.benchmark = True
        if self.__params.device.type != 'cpu':
            model(torch.zeros(1, 3, self.__params.imgsz, self.__params.imgsz).to(self.__params.device).type_as(next(model.parameters())))
        frame = torch.from_numpy(frame).to(self.__params.device)
        frame = frame.float()
        frame /= 255.0
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)
        frame = torch.transpose(frame, 2, 3)
        pred = model(frame, augment=False)[0]
        pred = non_max_suppression(pred, self.__params.conf_thres, max_det=self.__params.max_det)
        label = ""
        for i, det in enumerate(pred):
            img_shape = frame.shape[2:]
            detected_plate_shape = detected_plate.shape
            s_ = f'{i}: '
            s_ += '%gx%g ' % img_shape
            if len(det):
                gain = min(img_shape[0] / detected_plate_shape[0], img_shape[1] / detected_plate_shape[1])
                coords = det[:, :4]
                pad = (img_shape[1] - detected_plate_shape[1] * gain) / 2, (img_shape[0] - detected_plate_shape[0] * gain) / 2
                coords[:, [0, 2]] -= pad[0]
                coords[:, [1, 3]] -= pad[1]
                coords[:, :4] /= gain
                coords[:, 0].clamp_(0, detected_plate_shape[1])
                coords[:, 1].clamp_(0, detected_plate_shape[0])
                coords[:, 2].clamp_(0, detected_plate_shape[1])
                coords[:, 3].clamp_(0, detected_plate_shape[0])
                det[:, :4] = coords.round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s_ += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy, conf, cls in reversed(det):
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())
                    confidence_score = conf
                    class_index = cls
                    object_name = names[int(cls)]
                    if object_name.lower() == 'plate':
                        cropped_plate = frame[:, :, y1:y2, x1:x2].squeeze().permute(1, 2, 0).cpu().numpy()
                        cropped_plate = (cropped_plate * 255).astype(np.uint8)
                    c = int(cls)
                    label = names[c] if self.__params.hide_conf else f'{names[c]} {conf:.2f}'
                    tl = self.__params.rect_thickness
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(detected_plate, c1, c2, self.__params.color, thickness=tl, lineType=cv2.LINE_AA)
                    if label:
                        tf = max(tl - 1, 1)
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(detected_plate, c1, c2, self.__params.color, -1, cv2.LINE_AA)
                        cv2.putText(detected_plate, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return detected_plate, cropped_plate

    def __easyocr_model_load(self):
        text_reader = easyocr.Reader(["en"])
        return text_reader

    def __easyocr_model_works(self, cropped_image):
        texts = list()

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        
        results = self.__text_reader.readtext(edged)
        for (bbox, text, prob) in results:
            texts.append(text)
        return texts

    def __save_results(self, text, csv_filename, mutex):
        try:
            mutex.acquire()
            with open(csv_filename, mode="a", newline="") as f:
                csv_writer = csv.writer(f, delimiter="-", quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(text)
        finally:
            mutex.release()

    def __filter_plate_numbers(self, plate_numbers):
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        filtered = [''.join(filter(lambda x: x in valid_chars, plate)) for plate in plate_numbers]
        return filtered

    def start(self):
        self.__reader.start()

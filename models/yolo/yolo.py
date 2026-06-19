import colorsys
import os
import time
import dlib
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/best_epoch_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   所使用到的yolov8的版本：
        #   n : 对应yolov8_n
        #   s : 对应yolov8_s
        #   m : 对应yolov8_m
        #   l : 对应yolov8_l
        #   x : 对应yolov8_x
        #------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        print("self.num_classes:",self.num_classes)
        self.bbox_util                      = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        print("self.num_classes:",self.num_classes)
        self.net    = YoloBody(self.input_shape, self.num_classes, self.phi)
        
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        
        count=True  # 始终启用计数功能

        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image = cvtColor(image)  # 转换为RGB图像
        
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # 初始状态：记录每个目标的连续低置信度次数
        low_confidence_counter = {}  # 用于记录目标的连续低置信度次数
        confidence_threshold = 0.0  # 置信度阈值
        max_low_conf_count = 1      # 连续低置信度的最大帧数

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                                                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            
            if results[0] is None: 
                return image
            
            top_label = np.array(results[0][:, 5], dtype='int32')  # 类别
            top_conf = results[0][:, 4]  # 置信度
            top_boxes = results[0][:, :4]  # 边框

        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            seen_classes = set()  # 用来记录已经出现的类别
            for i in range(len(top_label)):
                class_id = top_label[i]
                if class_id not in seen_classes:
                    seen_classes.add(class_id)
                    num = np.sum(top_label == class_id)
                    if num > 0:
                        print(self.class_names[class_id], " : ", num)
                    classes_nums[class_id] = num
            print("classes_nums:", classes_nums)

        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        #---------------------------------------------------------#
        #   处理每个类别的目标，选择置信度最高的目标
        #---------------------------------------------------------#
        highest_confidence_per_class = {}  # 记录每个类别的置信度最高的目标
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            # 如果当前类别的目标还没有记录，或者当前目标的置信度更高，则更新记录
            if predicted_class not in highest_confidence_per_class:
                highest_confidence_per_class[predicted_class] = (score, box, i)
            else:
                if score > highest_confidence_per_class[predicted_class][0]:
                    highest_confidence_per_class[predicted_class] = (score, box, i)

        #---------------------------------------------------------#
        #   绘制最高置信度的目标
        #---------------------------------------------------------#
        for predicted_class, (score, box, i) in highest_confidence_per_class.items():
            # 绘制目标
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            # 计算目标中心点
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            # 绘制圆点
            radius = 10  # 圆点半径
            draw = ImageDraw.Draw(image)
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                        fill=(255, 0, 0), outline=(255, 0, 0))  # 红色圆点
            del draw

        return image, top_boxes, top_conf

    # def detect_image(self, image, crop=False, count=False):
    #     #---------------------------------------------------#
    #     #   计算输入图片的高和宽
    #     #---------------------------------------------------#
    #     image_shape = np.array(np.shape(image)[0:2])
    #     count=True
    #     #---------------------------------------------------------#
    #     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #     #---------------------------------------------------------#
    #     image = cvtColor(image)
        
    #     #---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     #---------------------------------------------------------#
    #     image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
    #     #---------------------------------------------------------#
    #     #   添加上batch_size维度
    #     #   h, w, 3 => 3, h, w => 1, 3, h, w
    #     #---------------------------------------------------------#
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    #     # 初始状态：记录每个目标的连续低置信度次数
    #     low_confidence_counter = {}  # 用于记录目标的连续低置信度次数
    #     confidence_threshold = 0.85  # 置信度阈值
    #     max_low_conf_count = 1      # 连续低置信度的最大帧数

    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()

    #         #---------------------------------------------------------#
    #         #   将图像输入网络当中进行预测！
    #         #---------------------------------------------------------#
    #         outputs = self.net(images)
    #         outputs = self.bbox_util.decode_box(outputs)
            
    #         #---------------------------------------------------------#
    #         #   将预测框进行堆叠，然后进行非极大抑制
    #         #---------------------------------------------------------#
    #         results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
    #                                                     image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            
    #         if results[0] is None: 
    #             return image

    #         top_label = np.array(results[0][:, 5], dtype='int32')  # 类别
    #         top_conf = results[0][:, 4]  # 置信度
    #         top_boxes = results[0][:, :4]  # 边框

    #     #---------------------------------------------------------#
    #     #   设置字体与边框厚度
    #     #---------------------------------------------------------#
    #     font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #     thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

    #     #---------------------------------------------------------#
    #     #   计数
    #     #---------------------------------------------------------#
    #     if count:
    #         print("top_label:", top_label)
    #         classes_nums = np.zeros([self.num_classes])
    #         for i in range(self.num_classes):
    #             num = np.sum(top_label == i)
    #             if num > 0:
    #                 print(self.class_names[i], " : ", num)
    #             classes_nums[i] = num
    #         print("classes_nums:", classes_nums)
        
    #     #---------------------------------------------------------#
    #     #   是否进行目标的裁剪
    #     #---------------------------------------------------------#
    #     if crop:
    #         for i, c in list(enumerate(top_boxes)):
    #             top, left, bottom, right = top_boxes[i]
    #             top = max(0, np.floor(top).astype('int32'))
    #             left = max(0, np.floor(left).astype('int32'))
    #             bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    #             right = min(image.size[0], np.floor(right).astype('int32'))
                
    #             dir_save_path = "img_crop"
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             crop_image = image.crop([left, top, right, bottom])
    #             crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
    #             print("save crop_" + str(i) + ".png to " + dir_save_path)

    #     #---------------------------------------------------------#
    #     #   图像绘制
    #     #---------------------------------------------------------#
    #     for i, c in list(enumerate(top_label)):
    #         predicted_class = self.class_names[int(c)]
    #         box = top_boxes[i]
    #         score = top_conf[i]

    #         # 为每个目标添加唯一ID，用于连续低置信度计数
    #         # 将目标的（位置、类别）作为唯一标识符
    #         target_id = f"{predicted_class}_{i}"

    #         if target_id not in low_confidence_counter:
    #             low_confidence_counter[target_id] = 0

    #         if score < confidence_threshold:
    #             low_confidence_counter[target_id] += 1
    #         else:
    #             low_confidence_counter[target_id] = 0

    #         # 如果连续3次低置信度，则跳过该目标
    #         if low_confidence_counter[target_id] >= max_low_conf_count:
    #             print("----------------------------------------------------------")
    #             continue
    #         print(score)
    #         top, left, bottom, right = box
    #         top = max(0, np.floor(top).astype('int32'))
    #         left = max(0, np.floor(left).astype('int32'))
    #         bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    #         right = min(image.size[0], np.floor(right).astype('int32'))

    #         # 计算目标中心点
    #         center_x = (left + right) // 2
    #         center_y = (top + bottom) // 2

    #         # 绘制圆点
    #         radius = 10  # 圆点半径
    #         draw = ImageDraw.Draw(image)
    #         draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], 
    #                     fill=(255, 0, 0), outline=(255, 0, 0))  # 红色圆点
    #         del draw

    #     return image, top_boxes, top_conf

    # def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            # 计算目标中心点
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            # 绘制圆点，移除标签
            radius = 10  # 圆点半径，可以根据需要调整
            draw = ImageDraw.Draw(image)
            # draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], 
            #             fill=self.colors[c], outline=self.colors[c])
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], 
                     fill=(255, 0, 0), outline=(255, 0, 0))  # 红色圆点
            # 删除draw对象
            del draw
            # # 计算透明度，使用帧计数器来控制透明度变化
            # self.frame_counter += 1
            # alpha = int((self.max_alpha - self.min_alpha) / 2 * (1 + np.sin(self.frame_counter * 2 * np.pi / self.blink_interval)) + self.min_alpha)

            # # 处理每一帧时，动态绘制圆点
            # for i, c in enumerate(top_label):
            #     predicted_class = self.class_names[int(c)]
            #     box = top_boxes[i]
            #     score = top_conf[i]

            #     top, left, bottom, right = box

            #     top = max(0, np.floor(top).astype('int32'))
            #     left = max(0, np.floor(left).astype('int32'))
            #     bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            #     right = min(image.size[0], np.floor(right).astype('int32'))

            #     # 计算目标中心点
            #     center_x = (left + right) // 2
            #     center_y = (top + bottom) // 2

            #     # 绘制圆点，使用当前透明度值
            #     draw = ImageDraw.Draw(image)
            #     draw.ellipse([center_x - self.radius, center_y - self.radius, center_x + self.radius, center_y + self.radius],
            #                 fill=(255, 0, 0, alpha), outline=(255, 0, 0))  # 红色圆点，带透明度

            #     del draw
            # label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # # label_size = draw.textsize(label, font)
            # label_size = draw.textbbox((0, 0), label, font)
            # label_width = label_size[2] - label_size[0]
            # label_height = label_size[3] - label_size[1]

            # label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            # rect = dlib.rectangle(top, left, bottom, right)

            # if top - label_height >= 0:
            #     text_origin = np.array([left, top - label_height])
            # else:
            #     text_origin = np.array([left, top + 1])

            # for i in range(thickness):
            #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + [label_width, label_height])], fill=self.colors[c])
            # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            # del draw

        return image,top_boxes,score
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])

        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
        #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
        #     draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        #     del draw

        # return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            dbox, cls, x, anchors, strides = self.net(images)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score      = np.max(sigmoid(sub_output[..., :]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

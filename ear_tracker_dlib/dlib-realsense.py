import dlib
import cv2
import pyrealsense2 as rs
import numpy as np
import json
from file import File

file = File()
f = open("file.txt","w")


def xy_mean(name):
    x = []
    y = []
    for n in name:
        x.append(landmarks.part(n).x)
        y.append(landmarks.part(n).y)
    x = int(np.mean(x))
    y = int(np.mean(y))
    return x, y


# def image_processing(frame):
#     kernel = np.ones((3,3),np.uint8)
#     new_frame = cv2.bilateralFilter(frame, 10, 15, 15)
 #   new_frame = cv2.erode(new_frame, kernel, iterations=3)


#
# def get_aligned_images():
#     frames = pipeline.wait_for_frames()  # 等待获取图像帧
#     aligned_frames = align.process(frames)  # 获取对齐帧
#     aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
#     color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
#
#     ############### 相机参数的获取 #######################
#     intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
#     depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
#     camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
#                          'ppx': intr.ppx, 'ppy': intr.ppy,
#                          'height': intr.height, 'width': intr.width,
#                          'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
#                          }
#     # 保存内参到本地
#     with open('./intrinsics.json', 'w') as fp:
#         json.dump(camera_parameters, fp)
#     #######################################################
#
#     depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
#     depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
#     depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
#     color_image = np.asanyarray(color_frame.get_data())  # RGB图
#
#     # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
#     return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


class File():
    def file(self,text):
        f=open("file.txt","a")
        f.write(text)



if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    profile = pipeline.start(config)
    align_to_color = rs.align(rs.stream.color) # align

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    color_green = (0, 255, 0)
    line_width = 3
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            # frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            # depth_frame = frames.get_depth_frame()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ############### 相机参数的获取 #######################
            intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                                 'ppx': intr.ppx, 'ppy': intr.ppy,
                                 'height': intr.height, 'width': intr.width,
                                 'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                                 }
            # 保存内参到本地
            with open('./intrinsics.json', 'w') as fp:
                json.dump(camera_parameters, fp)
            #######################################################

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            mask = np.zeros([color_image.shape[0], color_image.shape[1]], dtype=np.uint8)
            mask[0:480, 0:848] = 255

            if not depth_frame or not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            # img = image_processing(img)
            # 检测
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_img,1)
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                landmarks = predictor(gray_img, face)
                left_eyebrow = [18, 19, 20, 21, 22]
                right_eyebrow = [23, 24, 25, 26, 27]
                left_eye = [37, 40]
                right_eye = [43, 46]
                nose = [28, 29, 30, 31, 32, 33, 34, 35, 36]
                mouse = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
                # left_ear = [0, 1, 2]
                # right_ear = [14, 15]
                left_ear = [0, 1, 2, 3, 4]
                right_ear = [12, 13, 14, 15, 16]

                tmp = 0
                text = ''
                for n in [left_ear, right_ear, nose]: #
                    x, y = xy_mean(n)

                    depth = round(depth_frame.get_distance(x,y),2)
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    [a, b, c] = camera_coordinate
                    a = round(a, 3)
                    b = round(b, 3)
                    c = round(c, 3)

                    classes = ['left_ear', 'right_ear', 'nose']
                    text_tmp = '%s:%s'%(classes[tmp],[a,b,c])
                    text = text+'  '+text_tmp
                    cv2.putText(img, text, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                    cv2.circle(img, (x, y), 1, (0, 255, 0), thickness=1)
                    tmp += 1
                    # if tmp == 1:
                    #     tmp_text = text
                    if tmp == 3:
                        tmp_text = str(text)
                        file.file(tmp_text+"\n") # 耳朵的真实三维坐标存储到txt文件



            cv2.imshow("face", img)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

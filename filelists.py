import os
import cv2

# 指定文件夹路径
folder_path = "D:\data\data_root"
dest_path="D:\wav2lip_hd\filelists"

# 遍历文件夹中的所有 AVI 文件
avi_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]

# 计算所有 AVI 文件的总帧数
total_frames = sum([cv2.VideoCapture(os.path.join(folder_path, f)).get(cv2.CAP_PROP_FRAME_COUNT) for f in avi_files])
min=total_frames/25/60
print('总帧率为{},折合时长为{}分钟,即{}小时'.format(total_frames,min,min/60))
print('写入中....................................................................')
sum_frames=0
# 遍历每个 AVI 文件，将文件名写入 train.txt 或 val.txt 文件
i=0
train_frames=[]
for avi_file in avi_files:
    video_path = os.path.join(folder_path, avi_file)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sum_frames+=num_frames
    if sum_frames < 0.8 * total_frames:
        with open('D:\wav2lip_hd/filelists/train.txt', 'a') as f:
            index = avi_file.rfind('.')
            avi_file = avi_file[:index]
            f.write(avi_file + '\n')
            train_frames.append(sum_frames)
            i+=1
            # if(i==1):
            #     print(avi_file)
    else:
        with open('D:\wav2lip_hd/filelists/val.txt', 'a') as f:
            index = avi_file.rfind('.')
            avi_file = avi_file[:index]
            f.write(avi_file + '\n')

xunlianji=train_frames[i-1]
ceshiji=total_frames-xunlianji
print('已处理完毕：')
print('训练集总帧数为{}，折合时长{:.2f}小时。'.format(xunlianji, xunlianji/25/60/60))
print('测试集总帧数为{}，折合时长{:.2f}小时。'.format(ceshiji, ceshiji/25/60/60))
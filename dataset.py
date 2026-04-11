import torch
from torch.utils.data import Dataset
from modelscope import CLIPProcessor, CLIPModel
# from utils import *
import random
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import os
import shutil
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


class DatasetBuilder:
    """
    为 AnomalyRuler 构建新数据集的工具类。
    负责：视频抽帧、创建目录结构、生成 CSV 索引文件。

    使用示例（在项目根目录下运行 Python）：
    -------------------------------------------------------
    from dataset import DatasetBuilder

    builder = DatasetBuilder('Mydataset')

    # 第 1 步：查看目录结构（此时是空的）
    builder.show_structure()

    # 第 2 步：从视频抽帧到 train/（正常帧，用于规则归纳参考）
    builder.extract_frames(
        video_paths=['/path/to/normal_video.mp4'],
        target='train',           # 放入 train/ 目录
        sample_interval=30,       # 每 30 帧取 1 帧
    )

    # 第 3 步：从视频抽帧到 test/（测试帧）
    builder.extract_frames(
        video_paths=['/path/to/test_video1.mp4', '/path/to/test_video2.mp4'],
        target='test',
        sample_interval=30,
    )

    # 第 4 步：生成 train.csv（所有 train/ 下的帧，label=0）
    builder.create_train_csv()

    # 第 5 步：生成 test_frame/*.csv（每个视频一个 CSV）
    #   video_labels: 视频级标签，1=异常视频，0=正常视频
    #   当 video_labels=1 时，所有帧标记为 1
    builder.create_test_csvs(video_labels={'test_video1': 1, 'test_video2': 1})

    # 第 6 步：再次查看目录结构，确认一切就绪
    builder.show_structure()
    -------------------------------------------------------
    """

    def __init__(self, data_name):
        """
        初始化 DatasetBuilder。

        :param data_name: 数据集名称，会在项目根目录下创建同名文件夹
        """
        self.data_name = data_name
        self.root = data_name  # 相对于项目根目录
        # 创建所需的子目录
        for sub in ['train', 'test_frame', 'test_frame_description',
                     'modified_test_frame_description']:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
        print(f"[DatasetBuilder] 数据集根目录: {os.path.abspath(self.root)}")

    def extract_frames(self, video_paths, target='test', sample_interval=1):
        """
        从视频文件中抽帧，保存为 JPG 图片。

        :param video_paths: 视频文件路径列表
        :param target: 'train' 或 'test'，决定帧保存到哪个目录
        :param sample_interval: 采样间隔，1=每帧都取，30=每30帧取1帧
        """
        for vpath in video_paths:
            if not os.path.isfile(vpath):
                print(f"  [跳过] 文件不存在: {vpath}")
                continue

            video_name = os.path.splitext(os.path.basename(vpath))[0]
            if target == 'train':
                out_dir = os.path.join(self.root, 'train', video_name)
            else:
                # test 帧也放在 train/ 同级的结构中方便管理
                # 但路径会记录在 test_frame/*.csv 中
                out_dir = os.path.join(self.root, 'test', video_name)

            os.makedirs(out_dir, exist_ok=True)

            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            saved = 0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval == 0:
                    fname = f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(os.path.join(out_dir, fname), frame)
                    saved += 1
                frame_idx += 1

            cap.release()
            print(f"  [完成] {video_name}: 总帧数={total}, fps={fps:.1f}, "
                  f"采样间隔={sample_interval}, 保存了 {saved} 帧 → {out_dir}")

    def create_train_csv(self):
        """
        扫描 train/ 目录下所有图片，生成 train.csv（label 全为 0）。
        """
        train_dir = os.path.join(self.root, 'train')
        file_paths = sorted(glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True))
        if not file_paths:
            print(f"  [警告] {train_dir} 下没有找到 jpg 文件")
            return

        df = pd.DataFrame({'image_path': file_paths, 'label': [0] * len(file_paths)})
        csv_path = os.path.join(self.root, 'train.csv')
        df.to_csv(csv_path, index=False)
        print(f"  [完成] 生成 {csv_path}，共 {len(df)} 条记录（全部 label=0）")

    def create_test_csvs(self, normal_names=None, anomaly_names=None):
        """
        为 test/ 下的每个视频目录生成 test_frame/{video_name}.csv。

        :param normal_names: set，正常测试视频名集合 → label=0
        :param anomaly_names: set，异常测试视频名集合 → label=1
                              未在任何集合中的视频默认 label=1
        """
        normal_names = normal_names or set()
        anomaly_names = anomaly_names or set()

        test_dir = os.path.join(self.root, 'test')
        if not os.path.isdir(test_dir):
            print(f"  [错误] 目录不存在: {test_dir}，请先运行 extract_frames(target='test')")
            return

        video_dirs = sorted([d for d in os.listdir(test_dir)
                             if os.path.isdir(os.path.join(test_dir, d))])
        if not video_dirs:
            print(f"  [警告] {test_dir} 下没有视频目录，请先运行 extract_frames(target='test')")
            return

        for vname in video_dirs:
            vdir = os.path.join(test_dir, vname)
            frames = sorted(glob(os.path.join(vdir, '*.jpg')))
            if not frames:
                print(f"  [跳过] {vname}: 没有帧图片")
                continue

            label = 0 if vname in normal_names else 1
            tag = "正常" if label == 0 else "异常"
            df = pd.DataFrame({'image_path': frames, 'label': [label] * len(frames)})
            csv_path = os.path.join(self.root, 'test_frame', f'{vname}.csv')
            df.to_csv(csv_path, index=False)
            print(f"  [完成] {csv_path}: {len(df)} 帧, {tag}(label={label})")

    def show_structure(self):
        """打印当前数据集的目录结构和文件统计。"""
        print(f"\n{'='*50}")
        print(f"数据集: {self.data_name}")
        print(f"根目录: {os.path.abspath(self.root)}")
        print(f"{'='*50}")

        for sub in ['train', 'test', 'test_frame', 'test_frame_description',
                     'modified_test_frame_description']:
            sub_path = os.path.join(self.root, sub)
            if not os.path.isdir(sub_path):
                print(f"  {sub}/  (不存在)")
                continue

            # 统计子目录和文件
            subdirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
            files = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))]

            if subdirs:
                total_frames = 0
                for sd in sorted(subdirs):
                    n = len(os.listdir(os.path.join(sub_path, sd)))
                    total_frames += n
                    print(f"  {sub}/{sd}/  ({n} 个文件)")
                if len(subdirs) > 1:
                    print(f"  {sub}/ 合计: {len(subdirs)} 个子目录, {total_frames} 个文件")
            elif files:
                print(f"  {sub}/  ({len(files)} 个文件: {', '.join(sorted(files)[:5])}{'...' if len(files)>5 else ''})")
            else:
                print(f"  {sub}/  (空)")

        # 检查关键文件
        for f in ['train.csv']:
            fpath = os.path.join(self.root, f)
            if os.path.isfile(fpath):
                n = len(pd.read_csv(fpath))
                print(f"  {f}  ({n} 条记录)")
            else:
                print(f"  {f}  (未生成)")

        print(f"{'='*50}\n")

class UBNormal_VideoOrganizer:
    '''
    Usage:
    organizer = UBNormal_VideoOrganizer('data', 'data/train', 'data/test/normal', 'data/test/abnormal')
    organizer.organize_videos('normal_training_video_names.txt', 'normal_validation_video_names.txt', 'abnormal_validation_video_names.txt')
    '''
    def __init__(self, data_folder, train_folder, test_normal_folder, test_abnormal_folder):
        self.data_folder = data_folder
        self.train_folder = train_folder
        self.test_normal_folder = test_normal_folder
        self.test_abnormal_folder = test_abnormal_folder


    def read_video_list(self, file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]

    def copy_videos(self, video_list, destination_folder):
        for scene_folder in os.listdir(self.data_folder):
            scene_path = os.path.join(self.data_folder, scene_folder)
            if os.path.isdir(scene_path):
                for video in os.listdir(scene_path):
                    src = os.path.join(scene_path, video)
                    dst = os.path.join(destination_folder, video)
                    # Check if the file name without extension is in the list and if src and dst are not the same
                    if os.path.splitext(video)[0] in video_list and src != dst:
                        print(f"Copying {video} to {destination_folder}")
                        shutil.copy2(src, dst)
                    else:
                        print(f"Skipping {video}")

    def organize_videos(self, normal_train_file, normal_test_file, abnormal_test_file):
        normal_train_videos = self.read_video_list(normal_train_file)
        normal_test_videos = self.read_video_list(normal_test_file)
        abnormal_test_videos = self.read_video_list(abnormal_test_file)

        self.copy_videos(normal_train_videos, self.train_folder)
        self.copy_videos(normal_test_videos, self.test_normal_folder)
        self.copy_videos(abnormal_test_videos, self.test_abnormal_folder)


class UBNormal_VideoFrameExtractor:
    '''
    Usage:
    base_folders = ['data/test/normal', 'data/test/abnormal', 'data/train']
    extractor = UBNormal_VideoFrameExtractor(base_folders)
    extractor.process_all_folders()
    '''
    def __init__(self, base_folders):
        """
        Initialize the VideoFrameExtractor with base folders.
        :param base_folders: List of folders containing videos to process.
        """
        self.base_folders = base_folders

    @staticmethod
    def extract_frames(video_path, destination_folder):
        """
        Extract frames from a video and save them in a specified folder.
        :param video_path: Path to the video file.
        :param destination_folder: Folder where extracted frames will be saved.
        """
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_path = os.path.join(destination_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()

    def process_folder(self, folder_path):
        """
        Process all videos in a folder.
        :param folder_path: Path to the folder containing videos.
        """
        for video in os.listdir(folder_path):
            if video.endswith('.mp4'):
                video_path = os.path.join(folder_path, video)
                frame_folder = os.path.join(folder_path, os.path.splitext(video)[0])  # Remove .mp4
                self.extract_frames(video_path, frame_folder)

    def process_all_folders(self):
        """ Process all base folders. """
        for folder in self.base_folders:
            self.process_folder(folder)


class Label_loader_save:
    #get the ground truth label for 'ped2' and 'avenue'
    ### use example
    # gt_loader = Label_loader_save('ped2')  # Get gt labels.
    # gt = gt_loader()
    def __init__(self,name):
        self.name = name
        self.frame_path = f'{name}/test'
        self.mat_path = f'{name}/{name}.mat'
        video_folders = os.listdir(self.frame_path)
        video_folders.sort()
        self.video_folders = [os.path.join(self.frame_path, aa) for aa in video_folders]

    def __call__(self):
        data = self.load_ucsd_avenue()
        df = pd.DataFrame(data, columns=['image_path', 'label'])
        df.to_csv(f'{self.name}/test.csv', index=False)

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']
        all_data = []
        for i, folder in enumerate(self.video_folders):
            frame_files = sorted(os.listdir(folder))
            length = len(frame_files)
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]
                sub_video_gt[start: end] = 1

            for frame, label in zip(frame_files, sub_video_gt):
                all_data.append([os.path.join(folder, frame), label])

        return all_data


def create_train_csv(data_name):
    '''
    :param data_name: 'UBNormal', 'SHTech', 'ped2', 'avenue'
    :return:
    '''
    source_directory = f'{data_name}/train'
    file_paths = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    labels = list(np.zeros(len(file_paths), dtype=int))

    data = {'image_path': sorted(file_paths), 'label': labels}
    df = pd.DataFrame(data)
    csv_file_path = f'{data_name}/train.csv'
    df.to_csv(csv_file_path, index=False)


def create_test_UBNormal_csv(data_name):
    '''
    :param data_name: 'UBNormal'
    :return: create_test_UBNormal_csv('UBNormal')
    '''
    source_directory = f'{data_name}/test'
    file_paths = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # labels = list(np.load('SHTech/frame_labels_shanghai.npy'))
    labels = [1 if 'abnormal' in name else 0 for name in sorted(file_paths)]

    data = {'image_path': sorted(file_paths), 'label': labels}
    df = pd.DataFrame(data)
    csv_file_path = f'{data_name}/test.csv'
    df.to_csv(csv_file_path, index=False)


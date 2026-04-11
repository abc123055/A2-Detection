"""
一条命令构建 AnomalyRuler 所需的数据集目录结构。

目录组织要求:
    normal_videos/        ← 正常视频，一部分训练一部分测试
        video_a.mp4
        video_b.mp4
    anomaly_videos/       ← 异常视频，全部用于测试
        video_c.mp4
        video_d.mp4

用法示例:
    python build_dataset.py \
        --name Mydataset \
        --normal-dir /path/to/normal_videos \
        --anomaly-dir /path/to/anomaly_videos \
        --train-ratio 0.5 \
        --interval 30

生成结果:
    Mydataset/
    ├── train/                          (正常视频抽帧，label=0)
    ├── train.csv
    ├── test_frame/                     (每个测试视频一个 CSV)
    │   ├── video_b.csv                 (正常测试，label=0)
    │   ├── video_c.csv                 (异常测试，label=1)
    │   └── video_d.csv                 (异常测试，label=1)
    ├── test/
    ├── test_frame_description/         (后续 VLM 填充)
    └── modified_test_frame_description/
"""

import argparse
import os
from dataset import DatasetBuilder


def scan_videos(directory, exts=('.mp4', '.avi', '.mkv', '.mov')):
    """扫描目录下所有视频文件，返回路径列表。"""
    videos = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in exts:
            videos.append(os.path.join(directory, f))
    return videos


def parse_args():
    parser = argparse.ArgumentParser(
        description='为 AnomalyRuler 构建新数据集：抽帧 + 生成 CSV 索引',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--name', type=str, required=True,
                        help='数据集名称')
    parser.add_argument('--normal-dir', type=str, required=True,
                        help='正常视频目录')
    parser.add_argument('--anomaly-dir', type=str, required=True,
                        help='异常视频目录')
    parser.add_argument('--train-ratio', type=float, default=0.5,
                        help='正常视频中分配给训练集的比例 (默认: 0.5)')
    parser.add_argument('--interval', type=int, default=30,
                        help='采样间隔，每隔多少帧取 1 帧 (默认: 30)')
    return parser.parse_args()


def main():
    args = parse_args()

    normal_videos = scan_videos(args.normal_dir)
    anomaly_videos = scan_videos(args.anomaly_dir)
    print(f"正常视频: {len(normal_videos)} 个, 异常视频: {len(anomaly_videos)} 个")

    if not normal_videos:
        print(f"[错误] {args.normal_dir} 下没有找到视频文件")
        return
    if not anomaly_videos:
        print(f"[错误] {args.anomaly_dir} 下没有找到视频文件")
        return

    # 拆分正常视频 → 训练集 + 测试集
    split = max(1, int(len(normal_videos) * args.train_ratio))
    normal_train = normal_videos[:split]
    normal_test = normal_videos[split:]

    print(f"正常视频拆分: {len(normal_train)} 个训练 + {len(normal_test)} 个测试")
    print(f"异常视频: {len(anomaly_videos)} 个全部测试")

    builder = DatasetBuilder(args.name)

    # 1. 正常视频 → train/
    print("\n>>> 步骤 1/4: 正常视频抽帧 → train/")
    builder.extract_frames(normal_train, target='train', sample_interval=args.interval)

    # 2. 正常测试视频 + 异常视频 → test/
    print("\n>>> 步骤 2/4: 测试视频抽帧 → test/")
    if normal_test:
        builder.extract_frames(normal_test, target='test', sample_interval=args.interval)
    builder.extract_frames(anomaly_videos, target='test', sample_interval=args.interval)

    # 3. 生成 train.csv (label=0)
    print("\n>>> 步骤 3/4: 生成 train.csv")
    builder.create_train_csv()

    # 4. 生成 test_frame/*.csv
    #    正常测试视频 → label=0, 异常测试视频 → label=1
    print("\n>>> 步骤 4/4: 生成 test_frame/*.csv")
    normal_test_names = {os.path.splitext(os.path.basename(v))[0] for v in normal_test}
    anomaly_names = {os.path.splitext(os.path.basename(v))[0] for v in anomaly_videos}
    builder.create_test_csvs(normal_names=normal_test_names, anomaly_names=anomaly_names)

    builder.show_structure()


if __name__ == '__main__':
    main()

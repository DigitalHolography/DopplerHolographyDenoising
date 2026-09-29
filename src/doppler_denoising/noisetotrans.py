# -*- coding: utf-8 -*-
"""

Created on Wed Jun 24 14:51:49 2026



@author: solei



训练逻辑：

- 一个 dataset sample = 同一视频中连续 10 帧。

- 一个 batch = 2 条完整的 10 帧 sequence。

- 前 9 帧提供 bottleneck feature 历史。

- 最后一帧才是当前 target frame。

- 每次预测最后一帧时，temporal attention 看：

  [t-9, t-8, ..., t-2, t-1, t]

- 只有最后第 t 帧会被遮挡 block；每个 block 用自身平均亮度填满。

- 只有最后第 t 帧遮挡的位置会计算纯 L2 / MSE loss。

- target feature 作为 Query，前 9 帧 feature 作为 Key/Value。

- 在时间维做 scaled dot-product attention，再把结果交给 decoder。

- softmax 权重乘 1.1，使时间维权重总和为 1.1。

- 验证 sample 在训练开始前随机抽取一次，然后固定不变。

"""



from pathlib import Path

import cv2

import numpy as np

import winsound

from torch.utils.data import Dataset

import torch

import torch.nn as nn





# ============================================================
# 1. 放文件夹
# ============================================================
VIDEO_DIR = Path(

    r"C:\Users\Novovorontsovka\Downloads\video_masqued"

)





# 只接受 512×512 视频；其他尺寸直接跳过。
EXPECTED_HEIGHT = 512

EXPECTED_WIDTH = 512



# 最多只使用 200 个通过检查的有效视频。
# 注意：程序会先跳过空视频、非 512×512 视频和帧数不足的视频，
# 然后累计到 200 个有效视频后停止继续加载。
MAX_VALID_VIDEOS = 200





# ============================================================
# 2. 从文件名中取出视频编号
# ============================================================
def get_video_number(video_path):



    parts = video_path.stem.split("_")



    for part in reversed(parts):



        if part.isdigit():



            return int(part)



    raise RuntimeError(

        f"文件名里找不到视频编号：\n{video_path.name}"

    )





# ============================================================
# 3. 找到文件夹里的全部 AVI 视频，并按编号排序
# ============================================================
def find_all_avi_videos(video_dir):



    video_dir = Path(video_dir)



    # 只遍历目录一次，并用 suffix.lower() 同时兼容 .avi / .AVI。
    # 在 Windows 上分别 glob("*.avi") 和 glob("*.AVI") 会把同一个文件
    # 收集两次，导致同一 AVI 被重复读取。
    video_paths = [

        path

        for path in video_dir.iterdir()

        if path.is_file()

        and path.suffix.lower() == ".avi"

    ]



    # 按完整文件名排序。
    # 不再使用文件名里的 _7_p、_8_p 等局部编号排序，
    # 因为不同病人/不同采集的视频可能具有相同的局部编号。
    video_paths = sorted(

        video_paths,

        key=lambda path: path.name.lower()

    )



    return video_paths





# ============================================================
# 4. 读取一个视频，得到全部灰度帧
# ============================================================
def load_video_as_gray(video_path):



    cap = cv2.VideoCapture(

        str(video_path)

    )



    video_fps = float(

        cap.get(cv2.CAP_PROP_FPS)

    )



    if (

            not np.isfinite(video_fps)

            or video_fps <= 0

    ):



        video_fps = 25.0



    frames_gray = []



    while True:



        ret, frame_bgr = cap.read()



        if not ret:

            break



        frame_gray = cv2.cvtColor(

            frame_bgr,

            cv2.COLOR_BGR2GRAY

        )



        frames_gray.append(

            frame_gray

        )



    cap.release()



    if len(frames_gray) == 0:



        return (

            np.empty(

                (0, 0, 0),

                dtype=np.uint8

            ),

            video_fps

        )



    frames_gray = np.stack(

        frames_gray,

        axis=0

    )



    return frames_gray, video_fps





# ============================================================
# 4.2 建立有效圆形区域
# ============================================================
def create_circle_mask(H, W, cx=255, cy=255, r=260):



    valid_mask = np.zeros(

        (H, W),

        dtype=np.uint8

    )



    cv2.circle(

        valid_mask,

        (cx, cy),

        r,

        255,

        thickness=-1

    )



    return valid_mask





# ============================================================
# 5. 多视频 Dataset
# ============================================================
class MultiVideoDataset(Dataset):



    def __init__(

            self,

            video_paths,

            sequence_length

    ):



        super().__init__()



        self.video_paths = video_paths

        self.sequence_length = sequence_length



        self.all_frames = {}

        self.video_fps = {}

        self.video_paths_by_id = {}

        self.samples = []

        self.sample_indices_by_video = {}



        # video_id 是按完整文件列表自动分配的唯一整数。
        # 即使多个文件名都包含 _7_p，它们也会得到不同的 video_id，
        # 因而不会在 self.all_frames 等字典中互相覆盖。
        valid_video_count = 0



        for video_id, video_path in enumerate(self.video_paths):



            # 已经收集到 200 个有效视频后停止继续读取。
            if valid_video_count >= MAX_VALID_VIDEOS:

                print(

                    f"Reached MAX_VALID_VIDEOS={MAX_VALID_VIDEOS}. "

                    "Stop loading more videos."

                )

                break



            video_number = int(video_id)



            frames_gray, video_fps = load_video_as_gray(

                video_path

            )



            # 空视频直接跳过。
            if frames_gray.ndim != 3 or len(frames_gray) == 0:



                print(

                    f"SKIP video_id={video_number}: empty or unreadable -> "

                    f"{video_path.name}"

                )

                continue



            frame_height = int(frames_gray.shape[1])

            frame_width = int(frames_gray.shape[2])



            # 不是 512×512 的视频不进入 dataset。
            if (

                    frame_height != EXPECTED_HEIGHT

                    or frame_width != EXPECTED_WIDTH

            ):



                print(

                    f"SKIP video_id={video_number}: "

                    f"size={frame_width}x{frame_height}, "

                    f"expected={EXPECTED_WIDTH}x{EXPECTED_HEIGHT} -> "

                    f"{video_path.name}"

                )

                continue



            total_frames = len(frames_gray)



            # 至少要能组成一条完整 sequence。
            if total_frames < self.sequence_length:



                print(

                    f"SKIP video_id={video_number}: "

                    f"only {total_frames} frames, "

                    f"need at least {self.sequence_length} -> "

                    f"{video_path.name}"

                )

                continue



            self.all_frames[video_number] = frames_gray

            self.video_fps[video_number] = video_fps

            self.video_paths_by_id[video_number] = video_path



            print(

                f"video_id={video_number} | "

                f"file={video_path.name}"

            )



            valid_start_count = (

                total_frames

                - self.sequence_length

                + 1

            )



            for start_frame_index in range(

                    valid_start_count

            ):



                dataset_sample_index = len(self.samples)



                self.samples.append(

                    (

                        video_number,

                        start_frame_index

                    )

                )



                self.sample_indices_by_video.setdefault(

                    video_number,

                    []

                ).append(

                    dataset_sample_index

                )



            valid_video_count += 1



        if len(self.samples) == 0:



            raise RuntimeError(

                "没有任何满足条件的训练视频。\n"

                f"要求视频尺寸为 {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}，"

                f"且至少有 {self.sequence_length} 帧。"

            )



        print(

            f"Valid videos kept: {len(self.sample_indices_by_video)}"

            f"/{MAX_VALID_VIDEOS} | "

            f"Total sequence samples: {len(self.samples)}"

        )



    def __len__(self):



        return len(self.samples)



    def __getitem__(self, index):



        video_number, start_frame_index = self.samples[

            index

        ]



        frames_gray = self.all_frames[

            video_number

        ][

            start_frame_index:

            start_frame_index + self.sequence_length

        ]



        return (

            frames_gray,

            video_number,

            start_frame_index

        )





# ============================================================
# 一个 sample：
# 前 9 帧 history + 最后 1 帧 target
# ============================================================
HISTORY_FRAMES = 9



SEQUENCE_LENGTH = (

    HISTORY_FRAMES

    + 1

)



# 一个 batch 同时训练 2 条完整的 10 帧 sequence。
BATCH_SIZE = 2



video_paths = find_all_avi_videos(

    VIDEO_DIR

)



dataset = MultiVideoDataset(

    video_paths=video_paths,

    sequence_length=SEQUENCE_LENGTH

)





# ============================================================
# 6. Hyperparameters : patch hiding
# 每个 target 同时用自身平均亮度隐藏 32 个互不重叠的 32×32 patch。
# ============================================================
BLOCK_SIZE = 32

LOSS_BLOCK_SIZE = 32

NUMBER_OF_BLOCKS = 32





# ============================================================
# 7. patch hiding
# ============================================================
def replace_blocks_with_patch_mean(

        frames_gray,

        target_time_index,

        block_size,

        number_of_blocks,

        max_block_position_tries,

        cx,

        cy,

        r

):



    original_sequence = frames_gray.float() / 255.0



    # 当前最后一帧 target 的原始图像。
    target_frame = original_sequence[

        :,

        target_time_index

    ].clone()



    # 前 9 帧保持原样。
    # 最后一帧每个被选中的 block 用它自身的平均亮度填满。
    input_sequence = original_sequence.clone()



    block_mask = torch.zeros_like(

        target_frame

    )



    batch_size, _, height, width = (

        original_sequence.shape

    )



    circle_mask = create_circle_mask(

        height,

        width,

        cx,

        cy,

        r

    )



    loss_margin = (

        block_size

        - LOSS_BLOCK_SIZE

    ) // 2



    for batch_index in range(batch_size):



        for block_index in range(number_of_blocks):



            found_position = False



            for _ in range(

                    max_block_position_tries

            ):



                y0 = np.random.randint(

                    0,

                    height - block_size + 1

                )



                x0 = np.random.randint(

                    0,

                    width - block_size + 1

                )



                y1 = y0 + block_size

                x1 = x0 + block_size



                block_circle_mask = circle_mask[

                    y0:y1,

                    x0:x1

                ]



                if np.any(

                        block_circle_mask == 0

                ):

                    continue



                already_replaced = block_mask[

                    batch_index,

                    y0:y1,

                    x0:x1

                ].sum()



                if already_replaced > 0:

                    continue



                found_position = True



                break



            if not found_position:

                continue



            current_patch = original_sequence[

                batch_index,

                target_time_index,

                y0:y1,

                x0:x1

            ]



            patch_mean = current_patch.mean()



            # 用当前 patch 自己的平均亮度隐藏纹理和局部结构。
            input_sequence[

                batch_index,

                target_time_index,

                y0:y1,

                x0:x1

            ] = patch_mean



            block_mask[

                batch_index,

                y0 + loss_margin:y1 - loss_margin,

                x0 + loss_margin:x1 - loss_margin

            ] = 1.0



    return (

        input_sequence,

        target_frame,

        block_mask

    )





# ============================================================
# 8. 网络
# ============================================================
class ConvBlock(nn.Module):



    def __init__(

            self,

            in_channels,

            out_channels

    ):



        super().__init__()



        self.conv1 = nn.Conv2d(

            in_channels,

            out_channels,

            kernel_size=3,

            padding=1

        )



        self.norm1 = nn.GroupNorm(

            8,

            out_channels

        )



        self.act1 = nn.SiLU()



        self.conv2 = nn.Conv2d(

            out_channels,

            out_channels,

            kernel_size=3,

            padding=1

        )



        self.norm2 = nn.GroupNorm(

            8,

            out_channels

        )



        self.act2 = nn.SiLU()



    def forward(self, x):



        x = self.conv1(x)

        x = self.norm1(x)

        x = self.act1(x)



        x = self.conv2(x)

        x = self.norm2(x)

        x = self.act2(x)



        return x





class Downsample(nn.Module):



    def __init__(

            self,

            in_channels,

            out_channels

    ):



        super().__init__()



        self.down = nn.Conv2d(

            in_channels,

            out_channels,

            kernel_size=3,

            stride=2,

            padding=1

        )



    def forward(self, x):



        return self.down(x)





class Upsample(nn.Module):



    def __init__(

            self,

            in_channels,

            out_channels

    ):



        super().__init__()



        self.up = nn.ConvTranspose2d(

            in_channels,

            out_channels,

            kernel_size=2,

            stride=2

        )



    def forward(self, x):



        return self.up(x)





class TemporalFeatureAttention(nn.Module):



    def __init__(

            self,

            feature_channels,

            attention_channels,

            weight_scale

    ):



        super().__init__()



        if attention_channels <= 0:

            raise ValueError("attention_channels 必须大于 0。")



        if weight_scale <= 0:

            raise ValueError("weight_scale 必须大于 0。")



        self.attention_channels = int(attention_channels)

        self.scale = self.attention_channels ** -0.5

        self.weight_scale = float(weight_scale)



        # 轻量 1x1 Q/K/V projection；注意力只沿时间维计算。
        self.query_projection = nn.Conv2d(

            feature_channels,

            self.attention_channels,

            kernel_size=1

        )



        self.key_projection = nn.Conv2d(

            feature_channels,

            self.attention_channels,

            kernel_size=1

        )



        self.value_projection = nn.Conv2d(

            feature_channels,

            feature_channels,

            kernel_size=1

        )



        # 将 target feature 和历史 attention context 融回 512 channels。
        self.fusion = nn.Sequential(

            nn.Conv2d(

                2 * feature_channels,

                feature_channels,

                kernel_size=1

            ),

            nn.ReLU(inplace=True)

        )



    def forward(

            self,

            target_feature,

            history_features

    ):



        # 视频最前面的帧可能还没有历史；此时直接使用 target feature。
        if len(history_features) == 0:

            return target_feature



        # [B, T_history, C, H, W]
        stacked_history = torch.stack(

            history_features,

            dim=1

        )



        batch_size, history_length, channels, height, width = (

            stacked_history.shape

        )



        flattened_history = stacked_history.reshape(

            batch_size * history_length,

            channels,

            height,

            width

        )



        # Query 来自当前 target；Key/Value 只来自过去帧，不包含 target 自身。
        query = self.query_projection(

            target_feature

        ).unsqueeze(1)



        keys = self.key_projection(

            flattened_history

        ).reshape(

            batch_size,

            history_length,

            self.attention_channels,

            height,

            width

        )



        values = self.value_projection(

            flattened_history

        ).reshape(

            batch_size,

            history_length,

            channels,

            height,

            width

        )



        # 每个 16x16 空间位置独立做 temporal scaled dot-product attention。
        attention_scores = (

            query

            * keys

        ).sum(

            dim=2

        ) * self.scale



        attention_weights = torch.softmax(

            attention_scores,

            dim=1

        ).mul(

            self.weight_scale

        ).unsqueeze(2)



        history_context = (

            attention_weights

            * values

        ).sum(

            dim=1

        )



        fused_feature = self.fusion(

            torch.cat(

                [

                    target_feature,

                    history_context

                ],

                dim=1

            )

        )



        return fused_feature





class UNetTemporalAttention(nn.Module):



    def __init__(self):



        super().__init__()



        # 512 x 512
        self.enc1 = ConvBlock(

            1,

            32

        )



        # 256 x 256
        self.down1 = Downsample(

            32,

            64

        )



        self.enc2 = ConvBlock(

            64,

            64

        )



        # 128 x 128
        self.down2 = Downsample(

            64,

            128

        )



        self.enc3 = ConvBlock(

            128,

            128

        )



        # 64 x 64
        self.down3 = Downsample(

            128,

            256

        )



        self.enc4 = ConvBlock(

            256,

            256

        )



        # 32 x 32
        self.down4 = Downsample(

            256,

            512

        )



        self.enc5 = ConvBlock(

            512,

            512

        )



        # 16 x 16
        self.down5 = Downsample(

            512,

            512

        )



        self.enc6 = ConvBlock(

            512,

            512

        )



        self.bottleneck = ConvBlock(

            512,

            512

        )



        # 在最深层 16 x 16 bottleneck 做轻量 temporal Q/K/V attention。
        self.temporal_attention = TemporalFeatureAttention(

            feature_channels=512,

            attention_channels=ATTENTION_CHANNELS,

            weight_scale=ATTENTION_WEIGHT_SCALE

        )



        self.dec6 = ConvBlock(

            1024,

            512

        )



        self.up5 = Upsample(

            512,

            512

        )



        self.dec5 = ConvBlock(

            1024,

            512

        )



        self.up4 = Upsample(

            512,

            256

        )



        self.dec4 = ConvBlock(

            512,

            256

        )



        self.up3 = Upsample(

            256,

            128

        )



        self.dec3 = ConvBlock(

            256,

            128

        )



        self.up2 = Upsample(

            128,

            64

        )



        self.dec2 = ConvBlock(

            128,

            64

        )



        self.up1 = Upsample(

            64,

            32

        )



        self.dec1 = ConvBlock(

            64,

            32

        )



        self.final_conv = nn.Conv2d(

            32,

            1,

            kernel_size=3,

            padding=1

        )



    def encode_frame(self, input_frame):



        skip1 = self.enc1(

            input_frame

        )



        x = self.down1(

            skip1

        )



        skip2 = self.enc2(

            x

        )



        x = self.down2(

            skip2

        )



        skip3 = self.enc3(

            x

        )



        x = self.down3(

            skip3

        )



        skip4 = self.enc4(

            x

        )



        x = self.down4(

            skip4

        )



        skip5 = self.enc5(

            x

        )



        x = self.down5(

            skip5

        )



        skip6 = self.enc6(

            x

        )



        bottleneck_feature = self.bottleneck(

            skip6

        )



        return bottleneck_feature, (

            skip1,

            skip2,

            skip3,

            skip4,

            skip5,

            skip6

        )



    def decode_target(

            self,

            input_frame,

            fused_feature,

            target_skips

    ):



        (

            skip1,

            skip2,

            skip3,

            skip4,

            skip5,

            skip6

        ) = target_skips



        x = torch.cat(

            [

                fused_feature,

                skip6

            ],

            dim=1

        )



        x = self.dec6(x)



        x = self.up5(x)



        x = torch.cat(

            [x, skip5],

            dim=1

        )



        x = self.dec5(x)



        x = self.up4(x)



        x = torch.cat(

            [x, skip4],

            dim=1

        )



        x = self.dec4(x)



        x = self.up3(x)



        x = torch.cat(

            [x, skip3],

            dim=1

        )



        x = self.dec3(x)



        x = self.up2(x)



        x = torch.cat(

            [x, skip2],

            dim=1

        )



        x = self.dec2(x)



        x = self.up1(x)



        x = torch.cat(

            [x, skip1],

            dim=1

        )



        x = self.dec1(x)



        predicted_noise = self.final_conv(

            x

        )



        denoised = (

            input_frame

            - predicted_noise

        )



        return denoised



    def forward(self, input_sequence):



        # 支持 [B, T, H, W] 或 [B, T, 1, H, W]。
        if input_sequence.ndim == 4:



            input_sequence = input_sequence.unsqueeze(

                2

            )



        if (

                input_sequence.ndim != 5

                or input_sequence.shape[2] != 1

        ):



            raise ValueError(

                "input_sequence 必须是 [B,T,H,W] "

                "或 [B,T,1,H,W]。"

            )



        sequence_length = int(

            input_sequence.shape[1]

        )



        if sequence_length < 1:

            raise ValueError("sequence 至少需要 1 帧。")



        history_features = []

        target_frame = None

        target_feature = None

        target_skips = None



        # 所有帧共享 encoder；历史帧只提取 bottleneck，不运行 decoder。
        for time_index in range(sequence_length):



            current_frame = input_sequence[

                :,

                time_index

            ]



            current_feature, current_skips = self.encode_frame(

                current_frame

            )



            if time_index < sequence_length - 1:



                history_features.append(

                    current_feature

                )



            else:



                target_frame = current_frame

                target_feature = current_feature

                target_skips = current_skips



        fused_feature = self.temporal_attention(

            target_feature=target_feature,

            history_features=history_features

        )



        # 只使用 target skips 解码一次。
        return self.decode_target(

            input_frame=target_frame,

            fused_feature=fused_feature,

            target_skips=target_skips

        )





# ============================================================
# 9. Loss：仅 masked L2 / MSE
# ============================================================
def block_reconstruction_loss(

        prediction,

        target,

        block_mask

):



    """

    只在最后一帧被隐藏的 patch 内计算纯 L2（均方误差）。

    不使用区域权重、梯度项、Fair 正则或任何其他附加项。

    """



    squared_error = (

        prediction

        - target

    ).pow(2)



    masked_pixel_count_per_sample = block_mask.sum(

        dim=(1, 2, 3)

    )



    l2_loss_per_sample = (

        squared_error

        * block_mask

    ).sum(

        dim=(1, 2, 3)

    ) / masked_pixel_count_per_sample.clamp_min(1.0)



    batch_loss = l2_loss_per_sample.mean()



    return (

        batch_loss,

        l2_loss_per_sample,

        masked_pixel_count_per_sample

    )





# ============================================================
# 10. 训练配置与初始化
# ============================================================
DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)



print(

    "Using device:",

    DEVICE

)



if DEVICE.type == "cuda":



    print(

        "GPU:",

        torch.cuda.get_device_name(0)

    )



LEARNING_RATE = 5e-5

EPOCHS = 50



# 每个 epoch 抽取 3500 条 10-frame source sequence 来训练。
# 抽样时保证每个通过尺寸检查的视频至少贡献 1 条 sequence。
TRAIN_SAMPLES_PER_EPOCH = 3500



# 一开始随机选 400 条 source sequence 做 valid，之后固定不变。
VALID_SAMPLES = 400



EARLY_STOPPING_PATIENCE = 10

MAX_BLOCK_POSITION_TRIES = 100

SPLIT_RANDOM_SEED = 2026

VALIDATION_RANDOM_SEED = 10000

CX = 255

CY = 255

R = 260

ATTENTION_CHANNELS = 64

ATTENTION_WEIGHT_SCALE = 1.1



model = UNetTemporalAttention().to(

    DEVICE

)



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE

)





# ============================================================
# 11. 手动取一个 batch
# ============================================================
def get_one_sequence_batch(

        dataset,

        sample_indices

):



    batch_frames_gray = []

    batch_video_numbers = []

    batch_start_frame_indices = []



    for sample_index in sample_indices:



        frames_gray, video_number, start_frame_index = dataset[

            int(sample_index)

        ]



        batch_frames_gray.append(

            frames_gray

        )



        batch_video_numbers.append(

            video_number

        )



        batch_start_frame_indices.append(

            start_frame_index

        )



    frames_gray = torch.from_numpy(

        np.stack(

            batch_frames_gray,

            axis=0

        )

    )



    video_numbers = torch.tensor(

        batch_video_numbers,

        dtype=torch.long

    )



    start_frame_indices = torch.tensor(

        batch_start_frame_indices,

        dtype=torch.long

    )



    return (

        frames_gray,

        video_numbers,

        start_frame_indices

    )









# ============================================================
# 12. 随机固定 train / valid split
# ============================================================
def create_train_valid_indices(

        dataset,

        valid_samples,

        split_random_seed

):



    all_sample_indices = np.arange(

        len(dataset),

        dtype=np.int64

    )



    random_generator = np.random.default_rng(

        split_random_seed

    )



    # 为每个有效视频固定保留至少 1 条 train sequence。
    reserved_train_indices = []



    for video_number in sorted(

            dataset.sample_indices_by_video.keys()

    ):



        video_sample_indices = np.asarray(

            dataset.sample_indices_by_video[video_number],

            dtype=np.int64

        )



        reserved_train_indices.append(

            int(random_generator.choice(video_sample_indices))

        )



    reserved_train_indices = np.asarray(

        reserved_train_indices,

        dtype=np.int64

    )



    valid_candidates = np.setdiff1d(

        all_sample_indices,

        reserved_train_indices

    )



    if len(valid_candidates) < valid_samples:



        raise RuntimeError(

            "为每个视频保留至少 1 条 train sample 后，"

            "剩余 sample 不足以建立 validation set。\n"

            f"valid candidates = {len(valid_candidates)}\n"

            f"requested valid samples = {valid_samples}"

        )



    valid_sample_indices = random_generator.choice(

        valid_candidates,

        size=valid_samples,

        replace=False

    )



    train_sample_indices = np.setdiff1d(

        all_sample_indices,

        valid_sample_indices

    )



    return (

        train_sample_indices,

        valid_sample_indices

    )





# ============================================================
# 13. 前 9 帧作为 Key/Value + 当前帧作为 Query
# ============================================================
def run_attention_window(

        model,

        input_sequence,

        target_time_index,

        history_frames

):



    window_start_index = max(

        0,

        target_time_index - history_frames

    )



    attention_sequence = input_sequence[

        :,

        window_start_index:

        target_time_index + 1

    ]



    return model(

        attention_sequence

    )





# ============================================================
# 14. 训练一个 batch：2 条完整的 10 帧 sequence
# ============================================================
def train_one_sequence(

        model,

        frames_gray,

        optimizer,

        sequence_length,

        history_frames,

        block_size,

        number_of_blocks,

        max_block_position_tries,

        cx,

        cy,

        r

):



    optimizer.zero_grad(

        set_to_none=True

    )



    # 只有最后一帧作为 target。
    target_time_index = (

        sequence_length

        - 1

    )



    input_sequence, target_frame, block_mask = (

        replace_blocks_with_patch_mean(

            frames_gray=frames_gray,

            target_time_index=target_time_index,

            block_size=block_size,

            number_of_blocks=number_of_blocks,

            max_block_position_tries=max_block_position_tries,

            cx=cx,

            cy=cy,

            r=r

        )

    )



    if block_mask.sum().item() == 0:

        return None



    input_sequence = input_sequence.to(

        DEVICE

    )



    target_frame = target_frame.to(

        DEVICE

    ).unsqueeze(1)



    block_mask = block_mask.to(

        DEVICE

    ).unsqueeze(1)



    denoised_frame = run_attention_window(

        model=model,

        input_sequence=input_sequence,

        target_time_index=target_time_index,

        history_frames=history_frames

    )



    (

        current_loss,

        l2_loss_per_sample,

        masked_pixels_per_sample

    ) = block_reconstruction_loss(

        prediction=denoised_frame,

        target=target_frame,

        block_mask=block_mask

    )



    current_loss.backward()



    optimizer.step()



    return (

        current_loss.item(),

        l2_loss_per_sample.detach().cpu().tolist(),

        masked_pixels_per_sample.detach().cpu().tolist()

    )





# ============================================================
# 15. 验证一个 batch 的完整 sequence
# ============================================================
def validate_one_sequence(

        model,

        frames_gray,

        sequence_length,

        history_frames,

        block_size,

        number_of_blocks,

        max_block_position_tries,

        cx,

        cy,

        r

):



    # 只有最后一帧作为 target。
    target_time_index = (

        sequence_length

        - 1

    )



    with torch.no_grad():



        input_sequence, target_frame, block_mask = (

            replace_blocks_with_patch_mean(

                frames_gray=frames_gray,

                target_time_index=target_time_index,

                block_size=block_size,

                number_of_blocks=number_of_blocks,

                max_block_position_tries=max_block_position_tries,

                cx=cx,

                cy=cy,

                r=r

            )

        )



        if block_mask.sum().item() == 0:

            return None



        input_sequence = input_sequence.to(

            DEVICE

        )



        target_frame = target_frame.to(

            DEVICE

        ).unsqueeze(1)



        block_mask = block_mask.to(

            DEVICE

        ).unsqueeze(1)



        denoised_frame = run_attention_window(

            model=model,

            input_sequence=input_sequence,

            target_time_index=target_time_index,

            history_frames=history_frames

        )



        (

            current_loss,

            l2_loss_per_sample,

            masked_pixels_per_sample

        ) = block_reconstruction_loss(

            prediction=denoised_frame,

            target=target_frame,

            block_mask=block_mask

        )



    return (

        current_loss.item(),

        l2_loss_per_sample.detach().cpu().tolist(),

        masked_pixels_per_sample.detach().cpu().tolist()

    )





# ============================================================
# 15.5 逐条 sample 的 masked L2 输出
# ============================================================
def print_one_sample_loss(

        split_name,

        epoch_number,

        total_epochs,

        batch_number,

        total_batches,

        sample_position,

        samples_in_batch,

        dataset_sample_index,

        video_number,

        start_frame_index,

        target_time_index,

        masked_pixels,

        l2_loss

):



    target_frame_number = int(start_frame_index) + int(target_time_index)



    print(

        f"{split_name} | epoch={epoch_number:03d}/{total_epochs:03d} | "

        f"batch={batch_number:04d}/{total_batches:04d} | "

        f"sample={sample_position:02d}/{samples_in_batch:02d} | "

        f"dataset_idx={int(dataset_sample_index)} | "

        f"video={int(video_number)} | "

        f"target_frame={target_frame_number} | "

        f"mask={int(masked_pixels)} | "

        f"masked_L2={float(l2_loss):.6f}",

        flush=True

    )





# ============================================================
# 15.8 分层抽样：每个有效视频至少取 1 条 sequence
# ============================================================
def select_train_samples_for_epoch(

        dataset,

        train_sample_indices,

        train_samples_per_epoch

):



    train_sample_indices = np.asarray(

        train_sample_indices,

        dtype=np.int64

    )



    train_index_set = set(

        train_sample_indices.tolist()

    )



    mandatory_indices = []



    # 每个有效视频先随机取 1 条仍属于 train split 的 sequence。
    for video_number in sorted(

            dataset.sample_indices_by_video.keys()

    ):



        candidates = [

            sample_index

            for sample_index in dataset.sample_indices_by_video[video_number]

            if sample_index in train_index_set

        ]



        if len(candidates) == 0:

            continue



        mandatory_indices.append(

            int(np.random.choice(candidates))

        )



    if len(mandatory_indices) > train_samples_per_epoch:



        raise RuntimeError(

            "有效视频数量大于每个 epoch 的 sample 数，"

            "无法保证每个视频至少出现一次。\n"

            f"valid videos with train samples = {len(mandatory_indices)}\n"

            f"requested samples = {train_samples_per_epoch}"

        )



    mandatory_set = set(mandatory_indices)



    remaining_candidates = np.asarray(

        [

            sample_index

            for sample_index in train_sample_indices

            if int(sample_index) not in mandatory_set

        ],

        dtype=np.int64

    )



    remaining_count = (

        train_samples_per_epoch

        - len(mandatory_indices)

    )



    if remaining_count > 0:



        # 可用 sequence 少于所需数量时允许重复抽样，
        # 这样每个 epoch 仍然固定得到 3500 个 sample。
        replace_remaining = (

            remaining_count > len(remaining_candidates)

        )



        sampling_pool = remaining_candidates



        if len(sampling_pool) == 0:

            sampling_pool = train_sample_indices

            replace_remaining = True



        remaining_indices = np.random.choice(

            sampling_pool,

            size=remaining_count,

            replace=replace_remaining

        )



        selected_indices = np.concatenate(

            [

                np.asarray(mandatory_indices, dtype=np.int64),

                remaining_indices.astype(np.int64)

            ]

        )



    else:



        selected_indices = np.asarray(

            mandatory_indices,

            dtype=np.int64

        )



    # 打乱顺序，避免每个 epoch 开头总是 mandatory samples。
    np.random.shuffle(selected_indices)



    return selected_indices





# ============================================================
# 15.9 每个 epoch 随机选择一个训练视频并保存完整去噪视频
# ============================================================
def save_epoch_denoised_video(

        model,

        dataset,

        video_number,

        epoch_number,

        output_dir,

        device,

        history_frames,

        cx,

        cy,

        r

):



    """

    直接复用 dataset 首次载入内存的原始帧，不重新读取 AVI。

    每个目标帧都按训练逻辑从空 feature history 开始，只使用它自己和前 9 帧。

    视频最前面的 9 帧使用当时已有的较短因果历史，不使用未来帧。

    """



    video_number = int(video_number)

    output_dir = Path(output_dir)



    if video_number not in dataset.all_frames:



        raise KeyError(

            f"Denoise preview video_id={video_number} 不在有效 dataset 中。"

        )



    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )



    frames_gray = dataset.all_frames[

        video_number

    ]



    source_video_path = dataset.video_paths_by_id[

        video_number

    ]



    video_fps = dataset.video_fps[

        video_number

    ]



    frame_height = int(frames_gray.shape[1])

    frame_width = int(frames_gray.shape[2])



    valid_circle = torch.from_numpy(

        create_circle_mask(

            frame_height,

            frame_width,

            cx,

            cy,

            r

        ) > 0

    ).to(

        device=device

    ).unsqueeze(

        0

    ).unsqueeze(

        0

    )



    output_path = output_dir / (

        f"epoch_{epoch_number:03d}_"

        f"{source_video_path.stem}_denoised.avi"

    )



    fourcc = cv2.VideoWriter_fourcc(

        *"MJPG"

    )



    writer = cv2.VideoWriter(

        str(output_path),

        fourcc,

        video_fps,

        (frame_width, frame_height),

        True

    )



    if not writer.isOpened():



        writer.release()



        raise RuntimeError(

            "无法创建每个 epoch 的去噪 AVI：\n"

            f"{output_path}"

        )



    was_training = model.training

    model.eval()



    print(

        f"Denoising preview video after epoch {epoch_number:03d}: "

        f"video_id={video_number} | file={source_video_path.name} | "

        f"frames={len(frames_gray)}"

    )



    try:



        with torch.inference_mode():



            for target_frame_index in range(

                    len(frames_gray)

            ):



                window_start_index = max(

                    0,

                    target_frame_index - history_frames

                )



                input_sequence = torch.from_numpy(

                    frames_gray[

                        window_start_index:

                        target_frame_index + 1

                    ]

                ).to(

                    device=device,

                    dtype=torch.float32

                ).div_(

                    255.0

                ).unsqueeze(

                    0

                )



                target_input = input_sequence[

                    :,

                    -1

                ].unsqueeze(

                    1

                )



                denoised_frame = model(

                    input_sequence

                )



                denoised_frame = torch.nan_to_num(

                    denoised_frame,

                    nan=0.0,

                    posinf=1.0,

                    neginf=0.0

                )



                # Loss 只监督有效圆内；圆外沿用原始 target，避免无约束边缘伪影。
                denoised_frame = torch.where(

                    valid_circle,

                    denoised_frame,

                    target_input

                )



                denoised_uint8 = denoised_frame.clamp(

                    0.0,

                    1.0

                ).mul(

                    255.0

                ).round().to(

                    torch.uint8

                ).squeeze(

                    0

                ).squeeze(

                    0

                ).cpu().numpy()



                # 写成三通道灰度画面，兼容更多 AVI 播放器和 MJPG 后端。
                denoised_bgr = cv2.cvtColor(

                    denoised_uint8,

                    cv2.COLOR_GRAY2BGR

                )



                writer.write(

                    denoised_bgr

                )



    finally:



        writer.release()

        model.train(was_training)



    print(

        "Epoch denoised video saved:",

        output_path

    )



    return output_path





# ============================================================
# 16. 总训练循环
# ============================================================
def train_model(

        model,

        dataset,

        optimizer,

        epochs,

        train_samples_per_epoch,

        valid_samples,

        sequence_length,

        history_frames,

        block_size,

        number_of_blocks,

        max_block_position_tries,

        cx,

        cy,

        r,

        denoised_video_dir

):



    train_sample_indices, valid_sample_indices = (

        create_train_valid_indices(

            dataset=dataset,

            valid_samples=valid_samples,

            split_random_seed=SPLIT_RANDOM_SEED

        )

    )



    # 只用于逐 epoch 保存模型；不会改变训练、loss 或验证逻辑。
    EPOCH_MODEL_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    # 只从 train split 中出现过的 video_id 里随机选择预览视频。
    train_video_ids = sorted(
        {
            int(dataset.samples[int(sample_index)][0])
            for sample_index in train_sample_indices
        }
    )

    if len(train_video_ids) == 0:
        raise RuntimeError(
            "train split 中没有可用于逐 epoch 去噪预览的视频。"
        )

    print(
        "Each epoch will randomly denoise 1 training video."
    )

    best_valid_loss = float("inf")

    best_model_state = None

    epochs_without_improvement = 0



    for epoch_index in range(epochs):



        model.train()



        selected_train_indices = select_train_samples_for_epoch(

            dataset=dataset,

            train_sample_indices=train_sample_indices,

            train_samples_per_epoch=train_samples_per_epoch

        )



        train_loss_sum = 0.0

        train_loss_count = 0



        # BATCH_SIZE = 2。
        # 每次训练两条：
        # 前 9 帧 history + 最后 1 帧 target。
        for batch_start_index in range(

                0,

                len(selected_train_indices),

                BATCH_SIZE

        ):



            batch_sample_indices = selected_train_indices[

                batch_start_index:

                batch_start_index + BATCH_SIZE

            ]



            if len(batch_sample_indices) != BATCH_SIZE:

                continue



            frames_gray, video_numbers, start_frame_indices = (

                get_one_sequence_batch(

                    dataset,

                    batch_sample_indices

                )

            )



            batch_result = train_one_sequence(

                model=model,

                frames_gray=frames_gray,

                optimizer=optimizer,

                sequence_length=sequence_length,

                history_frames=history_frames,

                block_size=block_size,

                number_of_blocks=number_of_blocks,

                max_block_position_tries=max_block_position_tries,

                cx=cx,

                cy=cy,

                r=r

            )



            if batch_result is None:

                continue



            (

                batch_total_loss,

                l2_losses,

                masked_pixel_counts

            ) = batch_result



            batch_number = (

                batch_start_index // BATCH_SIZE

                + 1

            )



            total_batches = (

                len(selected_train_indices)

                // BATCH_SIZE

            )



            for sample_position in range(

                    len(batch_sample_indices)

            ):



                print_one_sample_loss(

                    split_name="TRAIN",

                    epoch_number=epoch_index + 1,

                    total_epochs=epochs,

                    batch_number=batch_number,

                    total_batches=total_batches,

                    sample_position=sample_position + 1,

                    samples_in_batch=len(batch_sample_indices),

                    dataset_sample_index=batch_sample_indices[

                        sample_position

                    ],

                    video_number=video_numbers[

                        sample_position

                    ].item(),

                    start_frame_index=start_frame_indices[

                        sample_position

                    ].item(),

                    target_time_index=sequence_length - 1,

                    masked_pixels=masked_pixel_counts[

                        sample_position

                    ],

                    l2_loss=l2_losses[

                        sample_position

                    ]

                )



            train_loss_sum += batch_total_loss

            train_loss_count += 1



        average_train_loss = train_loss_sum / max(

            train_loss_count,

            1

        )



        model.eval()



        valid_loss_sum = 0.0

        valid_loss_count = 0



        for valid_order, sample_index in enumerate(

                valid_sample_indices

        ):



            # validation 的 block 位置固定。
            # 所以不同 epoch 的 valid loss 可以公平比较。
            random_state = np.random.get_state()



            np.random.seed(

                VALIDATION_RANDOM_SEED

                + valid_order

            )



            frames_gray, video_numbers, start_frame_indices = (

                get_one_sequence_batch(

                    dataset,

                    [sample_index]

                )

            )



            batch_result = validate_one_sequence(

                model=model,

                frames_gray=frames_gray,

                sequence_length=sequence_length,

                history_frames=history_frames,

                block_size=block_size,

                number_of_blocks=number_of_blocks,

                max_block_position_tries=max_block_position_tries,

                cx=cx,

                cy=cy,

                r=r

            )



            np.random.set_state(

                random_state

            )



            if batch_result is None:

                continue



            (

                batch_total_loss,

                l2_losses,

                masked_pixel_counts

            ) = batch_result



            print_one_sample_loss(

                split_name="VALID",

                epoch_number=epoch_index + 1,

                total_epochs=epochs,

                batch_number=valid_order + 1,

                total_batches=len(valid_sample_indices),

                sample_position=1,

                samples_in_batch=1,

                dataset_sample_index=sample_index,

                video_number=video_numbers[0].item(),

                start_frame_index=start_frame_indices[0].item(),

                target_time_index=sequence_length - 1,

                masked_pixels=masked_pixel_counts[0],

                l2_loss=l2_losses[0]

            )



            valid_loss_sum += batch_total_loss

            valid_loss_count += 1



        average_valid_loss = valid_loss_sum / max(

            valid_loss_count,

            1

        )



        if average_valid_loss < best_valid_loss:



            best_valid_loss = average_valid_loss



            best_model_state = {

                key: value.detach().cpu().clone()

                for key, value in model.state_dict().items()

            }



            epochs_without_improvement = 0



        else:



            epochs_without_improvement += 1



        print(

            f"Epoch {epoch_index+1:03d}/{epochs:03d} | "

            f"train={average_train_loss:.6f} | "

            f"valid={average_valid_loss:.6f} | "

            f"best={best_valid_loss:.6f} | "

            f"no_improve={epochs_without_improvement:02d}/"

            f"{EARLY_STOPPING_PATIENCE:02d}"

        )



        # 保存当前 epoch 结束时的模型。
        # 这里保存的是 model.state_dict()，用于之后单独推理或比较任意 epoch。
        epoch_model_path = EPOCH_MODEL_DIR / (

            f"epoch_{epoch_index + 1:03d}.pth"

        )



        torch.save(

            model.state_dict(),

            epoch_model_path

        )



        print(

            "Epoch model saved:",

            epoch_model_path

        )



        # 每个 epoch 结束后，从 train split 中随机选择一个视频完整去噪。
        random_preview_video_id = int(
            np.random.choice(train_video_ids)
        )

        print(
            "Random training video selected for epoch preview:",
            f"video_id={random_preview_video_id} |",
            dataset.video_paths_by_id[random_preview_video_id].name
        )

        save_epoch_denoised_video(
            model=model,
            dataset=dataset,
            video_number=random_preview_video_id,
            epoch_number=epoch_index + 1,
            output_dir=denoised_video_dir,
            device=DEVICE,
            history_frames=history_frames,
            cx=cx,
            cy=cy,
            r=r
        )



        winsound.Beep(

            1000,

            250

        )



        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:



            print(

                "Early stopping: validation loss has not improved for",

                EARLY_STOPPING_PATIENCE,

                "epochs."

            )



            break



    if best_model_state is None:



        raise RuntimeError(

            "训练中没有得到有效的 validation loss，无法保存模型。"

        )



    model.load_state_dict(

        best_model_state

    )



    MODEL_SAVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(

        model.state_dict(),

        MODEL_SAVE_PATH

    )



    print(

        "Best model saved:",

        MODEL_SAVE_PATH

    )



    return model





# ============================================================
# 17. 开始训练
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Novovorontsovka\Downloads\video_masqued"
)
MODEL_SAVE_PATH = (
    BASE_DIR
    / "runs"
    / "model"
    / "best_patchmean32_temporal_attention_l2_200videos_batch2_8000.pth"
)



# 每个 epoch 结束后保存当时模型的 state_dict。
# 之后可以直接拿任意一个 epoch_xxx.pth 做推理。
EPOCH_MODEL_DIR = (
    BASE_DIR
    / "runs"
    / "model"
    / "epoch_models_temporal_attention"
)



# 每个完成的 epoch 会随机选择一个训练视频，并在这里保存一条完整去噪 AVI。
DENOISED_VIDEO_DIR = (
    BASE_DIR
    / "runs"
    / "denoised_videos_temporal_attention"
)



model = train_model(

    model=model,

    dataset=dataset,

    optimizer=optimizer,

    epochs=EPOCHS,

    train_samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,

    valid_samples=VALID_SAMPLES,

    sequence_length=SEQUENCE_LENGTH,

    history_frames=HISTORY_FRAMES,

    block_size=BLOCK_SIZE,

    number_of_blocks=NUMBER_OF_BLOCKS,

    max_block_position_tries=MAX_BLOCK_POSITION_TRIES,

    cx=CX,

    cy=CY,

    r=R,


    denoised_video_dir=DENOISED_VIDEO_DIR

)   

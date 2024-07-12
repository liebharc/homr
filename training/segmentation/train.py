import os
import random
from collections.abc import Callable, Generator, Iterator
from multiprocessing import Process, Queue
from typing import Any, cast

import augly.image as imaugs  # type: ignore
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageColor, ImageEnhance

from homr.simple_logging import eprint
from homr.type_definitions import NDArray
from training.segmentation.build_label import build_label, close_lines
from training.segmentation.constant_min import CHANNEL_NUM
from training.segmentation.types import Model
from training.segmentation.unet import semantic_segmentation, u_net


def monkey_patch_float_for_imaugs() -> None:
    """
    Monkey path workaround of np.float for imaugs
    """
    np.float = float  # type: ignore # ruff: noqa: E731


def get_cvc_data_paths(dataset_path: str) -> list[list[str]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")

    dirs = [
        "curvature",
        "ideal",
        "interrupted",
        "kanungo",
        "rotated",
        "staffline-thickness-variation-v1",
        "staffline-thickness-variation-v2",
        "staffline-y-variation-v1",
        "staffline-y-variation-v2",
        "thickness-ratio",
        "typeset-emulation",
        "whitespeckles",
    ]

    data = []
    for dd in dirs:
        dir_path = os.path.join(dataset_path, dd)
        folders = os.listdir(dir_path)
        for folder in folders:
            data_path = os.path.join(dir_path, folder)
            imgs = os.listdir(os.path.join(data_path, "image"))
            for img in imgs:
                img_path = os.path.join(data_path, "image", img)
                staffline = os.path.join(data_path, "gt", img)
                symbol_path = os.path.join(data_path, "symbol", img)
                data.append([img_path, staffline, symbol_path])

    return data


def get_deep_score_data_paths(dataset_path: str) -> list[list[str]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")

    imgs = os.listdir(os.path.join(dataset_path, "images"))
    paths = []
    for img in imgs:
        image_path = os.path.join(dataset_path, "images", img)
        seg_path = os.path.join(dataset_path, "segmentation", img.replace(".png", "_seg.png"))
        paths.append([image_path, seg_path])
    return paths


def apply_gradient_contrast(
    image: Image.Image, start_contrast: float = 1.0, end_contrast: float = 0.7
) -> Image.Image:
    width, height = image.size
    gradient_array = np.linspace(start_contrast, end_contrast, num=width * height).reshape(
        (height, width)
    )
    gradient = Image.fromarray((gradient_array * 255).astype(np.uint8), mode="L")

    # Apply uniform contrast reduction
    contrast_factor = end_contrast
    enhancer = ImageEnhance.Contrast(image)
    contrast_reduced_image = enhancer.enhance(contrast_factor)

    # Blend the original image with the contrast-reduced image using the gradient mask
    blended_image = Image.composite(image, contrast_reduced_image, gradient)
    return blended_image


def preprocess_image(img_path: str, reduce_contrast: bool = False) -> Image.Image:
    image = Image.open(img_path).convert("1")

    if image.mode == "1":
        # The input image contains only one channel.
        arr = np.array(image)
        out = np.zeros(arr.shape + (3,), dtype=np.uint8)
        bg_is_white = np.count_nonzero(arr) > (arr.size * 0.7)
        bg_idx = np.where(arr == bg_is_white)

        # Change background color
        hue = random.randint(19, 60)
        sat = random.randint(0, 15)
        val = random.randint(70, 100)
        color = ImageColor.getrgb(f"hsv({hue}, {sat}%, {val}%)")
        out[bg_idx[0], bg_idx[1]] = color
        image = Image.fromarray(out)

    aug_image = image

    if reduce_contrast:
        # Reduce contrast randomly
        aug_image = apply_gradient_contrast(
            aug_image, random.uniform(0.3, 1.0), random.uniform(0.3, 1.0)
        )

    # Color jitter
    bright = (7 + random.randint(0, 6)) / 10  # 0.7~1.3
    saturation = (5 + random.randint(0, 7)) / 10  # 0.5~1.2
    contrast = (5 + random.randint(0, 10)) / 10  # 0.5~1.5
    aug_image = imaugs.color_jitter(
        aug_image, brightness_factor=bright, saturation_factor=saturation, contrast_factor=contrast
    )

    # Blur
    rad = random.choice(np.arange(0.1, 2.1, 0.5))
    aug_image = imaugs.blur(aug_image, radius=rad)

    # Pixel shuffle, kind of adding noise
    factor = random.choice(np.arange(0.1, 0.26, 0.05))
    aug_image = imaugs.shuffle_pixels(aug_image, factor=factor)

    # Image quality
    qa = random.randint(0, 100)
    aug_image = imaugs.encoding_quality(aug_image, quality=qa)

    # Pixelize (pretty similar to blur?)
    rat = random.randint(3, 10) / 10
    aug_image = imaugs.pixelization(aug_image, ratio=rat)

    return aug_image


def batch_transform(
    img: Image.Image | NDArray, trans_func: Callable[[Image.Image], Image.Image]
) -> NDArray:
    if isinstance(img, Image.Image):
        return np.array(trans_func(img))

    if not isinstance(img, np.ndarray):
        raise ValueError("Input image should be either PIL.Image or np.ndarray")
    color_channels = 3
    if len(img.shape) != color_channels:
        raise ValueError("Input image should be 3D array with shape (h, w, ch)")

    ch_num = img.shape[2]
    result = []
    for i in range(ch_num):
        tmp_img = Image.fromarray(img[..., i].astype(np.uint8))
        tmp_img = trans_func(tmp_img)
        result.append(np.array(tmp_img))
    return np.dstack(result)


class MultiprocessingDataLoader:
    def __init__(self, num_worker: int):
        self._queue: Queue[list[Any]] = Queue(maxsize=20)
        self._dist_queue: Queue[list[str]] = Queue(maxsize=30)
        self._process_pool = []
        for _ in range(num_worker):
            processor = Process(target=self._preprocess_image)
            processor.daemon = True
            self._process_pool.append(processor)
        self._pdist = Process(target=self._distribute_process)
        self._pdist.daemon = True

    def _start_processes(self) -> None:
        if not self._pdist.is_alive():
            self._pdist.start()
        for process in self._process_pool:
            if not process.is_alive():
                process.start()

    def _terminate_processes(self) -> None:
        self._pdist.terminate()
        for process in self._process_pool:
            process.terminate()

    def _distribute_process(self) -> None:
        pass

    def _preprocess_image(self) -> None:
        pass


class DataLoader(MultiprocessingDataLoader):
    def __init__(
        self,
        feature_files: list[list[str]],
        win_size: int = 256,
        num_samples: int = 100,
        min_step_size: float = 0.2,
        num_worker: int = 4,
    ):
        super().__init__(num_worker)
        self.feature_files = feature_files
        random.shuffle(self.feature_files)
        self.win_size = win_size
        self.num_samples = num_samples

        if isinstance(min_step_size, float):
            min_step_size = max(min(abs(min_step_size), 1), 0.01)
            self.min_step_size = round(win_size * min_step_size)
        else:
            self.min_step_size = max(min(abs(min_step_size), win_size), 2)

        self.file_idx = 0

    def _distribute_process(self) -> None:
        while True:
            paths = self.feature_files[self.file_idx]
            self._dist_queue.put(paths)
            self.file_idx += 1
            if self.file_idx == len(self.feature_files):
                random.shuffle(self.feature_files)
                self.file_idx = 0

    def _preprocess_image(self) -> None:
        while True:
            if not self._queue.full():
                inp_img_path, staff_img_path, symbol_img_path = self._dist_queue.get()

                # Preprocess image with transformations that won't change view.
                image = preprocess_image(inp_img_path, reduce_contrast=True)

                # Random resize
                ratio = random.choice(np.arange(0.2, 1.21, 0.1))
                tar_w = int(ratio * image.size[0])
                tar_h = int(ratio * image.size[1])
                image = imaugs.resize(image, width=tar_w, height=tar_h)
                staff_img_array = cv2.imread(staff_img_path)
                staff_img_array = cv2.cvtColor(staff_img_array, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                staff_img_array = close_lines(staff_img_array)
                staff_img = Image.fromarray(staff_img_array)
                staff_img = imaugs.resize(staff_img, width=tar_w, height=tar_h)
                staff_img = imaugs.resize(staff_img_path, width=tar_w, height=tar_h)
                symbol_img = imaugs.resize(symbol_img_path, width=tar_w, height=tar_h)

                # Random perspective transform
                seed = random.randint(0, 1000)
                monkey_patch_float_for_imaugs()
                random_rotation = random.uniform(-5, 5)

                def perspect_trans(
                    img: Image.Image, seed: int = seed, random_rotation: float = random_rotation
                ) -> Any:
                    rotated = img.rotate(random_rotation)
                    return imaugs.perspective_transform(rotated, seed=seed, sigma=70)

                image_trans = np.array(perspect_trans(image))  # RGB image
                staff_img_trans = np.array(perspect_trans(staff_img))  # 1-bit mask
                symbol_img_trans = np.array(perspect_trans(symbol_img))  # 1-bit mask
                staff_img_trans = np.where(staff_img_trans, 1, 0)
                symbol_img_trans = np.where(symbol_img_trans, 1, 0)

                self._queue.put([image_trans, staff_img_trans, symbol_img_trans, ratio])

    def __iter__(self) -> Iterator[tuple[NDArray, NDArray]]:
        samples = 0

        self._start_processes()

        while samples < self.num_samples:
            image, staff_img, symbol_img, ratio = self._queue.get()

            start_x, start_y = 0, 0
            max_y = image.shape[0] - self.win_size
            max_x = image.shape[1] - self.win_size
            while (start_x < max_x) and (start_y < max_y):
                y_range = range(start_y, start_y + self.win_size)
                x_range = range(start_x, start_x + self.win_size)
                index = np.ix_(y_range, x_range)
                # Can't use two 'range' inside the numpy array for indexing.
                # Details refer to the following:
                # https://stackoverflow.com/questions/30020143/indexing-slicing-a-2d-numpy-array-using-the-range-arange-function-as-the-argumen
                feat = image[index]
                staff = staff_img[index]
                symbol = symbol_img[index]
                neg = np.ones_like(staff) - staff - symbol
                label = np.stack([neg, staff, symbol], axis=-1)

                yield feat, label

                y_step = random.randint(
                    round(self.min_step_size * ratio), round(self.win_size * ratio)
                )
                x_step = random.randint(
                    round(self.min_step_size * ratio), round(self.win_size * ratio)
                )
                start_y = min(start_y + y_step, max_y)
                start_x = min(start_x + x_step, max_x)

        self._terminate_processes()

    def get_dataset(
        self, batch_size: int
    ) -> tf.data.Dataset[Generator[tuple[NDArray, NDArray], None, None]]:
        def gen_wrapper() -> Generator[tuple[NDArray, NDArray], None, None]:
            yield from self

        return (
            tf.data.Dataset.from_generator(
                gen_wrapper,
                output_signature=(
                    tf.TensorSpec(
                        shape=(self.win_size, self.win_size, 3), dtype=tf.uint8, name=None
                    ),
                    tf.TensorSpec(
                        shape=(self.win_size, self.win_size, 3), dtype=tf.float32, name=None
                    ),
                ),
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )


class DsDataLoader(MultiprocessingDataLoader):
    def __init__(
        self,
        feature_files: list[list[str]],
        win_size: int = 256,
        num_samples: int = 100,
        step_size: float = 0.5,
        num_worker: int = 4,
    ):
        super().__init__(num_worker)
        self.feature_files = feature_files
        random.shuffle(self.feature_files)
        self.win_size = win_size
        self.num_samples = num_samples

        if isinstance(step_size, float):
            step_size = max(abs(step_size), 0.01)
            self.step_size = round(win_size * step_size)
        else:
            self.step_size = max(abs(step_size), 2)

        self.file_idx = 0

    def _distribute_process(self) -> None:
        while True:
            paths = self.feature_files[self.file_idx]
            self._dist_queue.put(paths)
            self.file_idx += 1
            if self.file_idx == len(self.feature_files):
                random.shuffle(self.feature_files)
                self.file_idx = 0

    def _preprocess_image(self) -> None:
        while True:
            if not self._queue.full():
                inp_img_path, seg_img_path = self._dist_queue.get()

                # Preprocess image with transformations that won't change view.
                image = preprocess_image(inp_img_path)
                strengthen_channels = {
                    1: (5, 5),
                }
                label = build_label(seg_img_path, strenghten_channels=strengthen_channels)

                # Random resize
                ratio = random.choice(np.arange(0.2, 1.21, 0.1))
                tar_w = int(ratio * image.size[0])
                tar_h = int(ratio * image.size[1])

                def trans_func(img: Image.Image, width: int = tar_w, height: int = tar_h) -> Any:
                    return imaugs.resize(img, width=width, height=height)

                image_trans = batch_transform(image, trans_func)
                label_trans = batch_transform(label, trans_func)

                # Random perspective transform
                seed = random.randint(0, 1000)
                monkey_patch_float_for_imaugs()
                random_rotation = random.uniform(-5, 5)

                def perspect_trans(
                    img: Image.Image, seed: int = seed, random_rotation: float = random_rotation
                ) -> Any:
                    rotated = img.rotate(random_rotation)
                    return imaugs.perspective_transform(rotated, seed=seed, sigma=70)

                image_arr = np.array(batch_transform(image_trans, perspect_trans))  # RGB image
                label_arr = np.array(batch_transform(label_trans, perspect_trans))

                self._queue.put([image_arr, label_arr, ratio])

    def __iter__(self) -> Iterator[tuple[NDArray, NDArray]]:
        samples = 0

        self._start_processes()

        while samples < self.num_samples:
            image, label, ratio = self._queue.get()

            # Discard bottom spaces that has no contents.
            staff = label[..., 1]
            yidx, _ = np.where(staff > 0)
            if len(yidx) > 0:
                max_y = min(np.max(yidx) + 100, image.shape[0])
            else:
                max_y = image.shape[0]

            max_y = max_y - self.win_size
            max_x = image.shape[1] - self.win_size
            grid_x = range(0, max_x, round(self.step_size * ratio))
            grid_y = range(0, max_y, round(self.step_size * ratio))
            meshgrid = np.meshgrid(grid_x, grid_y, indexing="ij")
            coords = np.dstack(meshgrid).reshape(-1, 2)
            np.random.shuffle(coords)
            for start_x, start_y in coords:
                y_range = range(start_y, start_y + self.win_size)
                x_range = range(start_x, start_x + self.win_size)
                index = np.ix_(y_range, x_range)

                # Can't use two 'range' inside the numpy array for indexing.
                # Details refer to the following:
                # https://stackoverflow.com/questions/30020143/indexing-slicing-a-2d-numpy-array-using-the-range-arange-function-as-the-argumen
                feat = image[index]
                ll = label[index]
                yield feat, ll

        self._terminate_processes()

    def get_dataset(
        self, batch_size: int
    ) -> tf.data.Dataset[Generator[tuple[NDArray, NDArray], None, None]]:
        def gen_wrapper() -> Generator[tuple[NDArray, NDArray], None, None]:
            yield from self

        return (
            tf.data.Dataset.from_generator(
                gen_wrapper,
                output_signature=(
                    tf.TensorSpec(
                        shape=(self.win_size, self.win_size, 3), dtype=tf.uint8, name=None
                    ),
                    tf.TensorSpec(
                        shape=(self.win_size, self.win_size, CHANNEL_NUM),
                        dtype=tf.float32,
                        name=None,
                    ),
                ),
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )


class WarmUpLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr: float = 0.1,
        warm_up_steps: int = 1000,
        decay_step: int = 3000,
        decay_rate: float = 0.25,
        min_lr: float = 1e-8,
    ):
        self.init_lr = init_lr
        self.warm_up_steps = warm_up_steps
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.min_lr = min_lr

        self.warm_step_size = (init_lr - min_lr) / warm_up_steps

    def __call__(self, step: int | tf.Tensor) -> float | tf.Tensor:
        step = tf.cast(step, tf.float32)
        warm_lr = self.min_lr + self.warm_step_size * step

        offset = step - self.warm_up_steps
        cycle = offset // self.decay_step
        start_lr = self.init_lr * tf.pow(self.decay_rate, cycle)
        end_lr = start_lr * self.decay_rate
        step_size = (start_lr - end_lr) / self.decay_step
        lr = start_lr - (offset - cycle * self.decay_step) * step_size
        true_lr = tf.where(offset > 0, lr, warm_lr)
        result = tf.maximum(true_lr, self.min_lr)
        return cast(float | tf.Tensor, result)

    def get_config(self) -> dict[str, Any]:
        return {
            "warm_up_steps": self.warm_up_steps,
            "decay_step": self.decay_step,
            "decay_rate": self.decay_rate,
            "min_lr": self.min_lr,
        }


def train_model(
    dataset_path: str,
    train_val_split: float = 0.1,
    learning_rate: float = 5e-4,
    epochs: int = 15,
    steps: int = 1000,
    batch_size: int = 8,
    val_steps: int = 200,
    val_batch_size: int = 8,
    early_stop: int = 8,
    data_model: str = "segnet",
) -> Model:
    if data_model == "segnet":
        feat_files = get_deep_score_data_paths(dataset_path)
    else:
        feat_files = get_cvc_data_paths(dataset_path)
    random.shuffle(feat_files)
    split_idx = round(train_val_split * len(feat_files))
    train_files = feat_files[split_idx:]
    val_files = feat_files[:split_idx]

    eprint(f"Loading dataset. Train/validation: {len(train_files)}/{len(val_files)}")
    if data_model == "segnet":
        win_size = 288
        train_data = DsDataLoader(
            train_files, win_size=win_size, num_samples=epochs * steps * batch_size
        ).get_dataset(batch_size)
        val_data = DsDataLoader(
            val_files, win_size=win_size, num_samples=epochs * val_steps * val_batch_size
        ).get_dataset(val_batch_size)
        model = u_net(win_size=win_size, out_class=CHANNEL_NUM)
    else:
        win_size = 256
        train_data = DataLoader(
            train_files, win_size=win_size, num_samples=epochs * steps * batch_size
        ).get_dataset(batch_size)
        val_data = DataLoader(
            val_files, win_size=win_size, num_samples=epochs * val_steps * val_batch_size
        ).get_dataset(val_batch_size)
        model = semantic_segmentation(win_size=256, out_class=3)

    eprint("Initializing model")
    optim = tf.keras.optimizers.Adam(learning_rate=WarmUpLearningRate(learning_rate))
    loss = tf.keras.losses.CategoricalFocalCrossentropy()
    model.compile(optimizer=optim, loss=loss, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=early_stop, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(
            "seg_unet.keras", save_weights_only=False, monitor="val_accuracy"
        ),
    ]

    eprint("Start training")
    try:
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            callbacks=callbacks,
        )
        return model
    except Exception as e:
        eprint(e)
        return model

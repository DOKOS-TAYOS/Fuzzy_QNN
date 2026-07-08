from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader
    train_dataset: TensorDataset
    val_dataset: TensorDataset | None
    test_dataset: TensorDataset
    metadata: dict[str, Any]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray | None
    y_val: np.ndarray | None
    x_test: np.ndarray
    y_test: np.ndarray


def build_dataloaders(config: ExperimentConfig) -> DataBundle:
    features, targets, class_names, feature_names = _load_dataset(config)
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = _split_dataset(
        features=features,
        targets=targets,
        test_size=config.dataset.test_size,
        val_size=config.dataset.val_size,
        seed=config.seed,
    )
    x_train_processed, x_val_processed, x_test_processed = _fit_transform_features(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        scale=config.dataset.scale,
        feature_reduction=config.dataset.feature_reduction,
        d_in=config.dataset.d_in,
        seed=config.seed,
    )

    train_dataset = _make_tensor_dataset(x_train_processed, y_train)
    val_dataset = (
        _make_tensor_dataset(x_val_processed, y_val)
        if x_val_processed is not None and y_val is not None
        else None
    )
    test_dataset = _make_tensor_dataset(x_test_processed, y_test)

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    metadata = {
        "d_in": int(x_train_processed.shape[1]),
        "n_classes": len(np.unique(targets)),
        "class_names": class_names,
        "feature_names": feature_names,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset) if val_dataset is not None else 0,
        "n_test": len(test_dataset),
        "train_fraction": 1.0 - config.dataset.test_size - config.dataset.val_size,
        "val_fraction": config.dataset.val_size,
        "test_fraction": config.dataset.test_size,
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
        x_train=x_train_processed,
        y_train=y_train,
        x_val=x_val_processed,
        y_val=y_val,
        x_test=x_test_processed,
        y_test=y_test,
    )


def make_noisy_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    noise_std: float,
    seed: int,
) -> DataLoader:
    features, targets = dataset.tensors
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(features.shape, generator=generator, dtype=features.dtype) * noise_std
    noisy_features = torch.clamp(features + noise, 0.0, 1.0)
    noisy_dataset = TensorDataset(noisy_features, targets.clone())
    return DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)


def _load_dataset(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    dataset_name = config.dataset.name.lower()
    if dataset_name == "iris":
        dataset = load_iris()
        return dataset.data, dataset.target, dataset.target_names.tolist(), dataset.feature_names
    if dataset_name == "breast_cancer":
        dataset = load_breast_cancer()
        return (
            dataset.data,
            dataset.target,
            dataset.target_names.tolist(),
            dataset.feature_names.tolist(),
        )
    if dataset_name == "wine":
        dataset = load_wine()
        return dataset.data, dataset.target, dataset.target_names.tolist(), dataset.feature_names
    if dataset_name == "synthetic":
        n_classes = config.dataset.n_classes or 2
        n_features = config.dataset.d_in
        features, targets = make_classification(
            n_samples=config.dataset.n_samples or 2000,
            n_features=n_features,
            n_informative=config.dataset.n_informative or max(2, n_features - 1),
            n_redundant=config.dataset.n_redundant or 0,
            n_classes=n_classes,
            class_sep=config.dataset.class_sep or 1.0,
            flip_y=config.dataset.flip_y or 0.03,
            random_state=config.seed,
        )
        class_names = [f"class_{index}" for index in range(n_classes)]
        feature_names = [f"feature_{index}" for index in range(n_features)]
        return features, targets, class_names, feature_names
    raise ValueError(f"Unsupported dataset '{config.dataset.name}'.")


def _split_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=seed,
    )
    if val_size <= 0.0:
        return x_train_val, y_train_val, None, None, x_test, y_test

    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=seed,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def _fit_transform_features(
    x_train: np.ndarray,
    x_val: np.ndarray | None,
    x_test: np.ndarray,
    scale: str,
    feature_reduction: str,
    d_in: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    train_working = x_train
    val_working = x_val
    test_working = x_test

    if scale == "standard_then_minmax":
        standard_scaler = StandardScaler()
        train_working = standard_scaler.fit_transform(train_working)
        if val_working is not None:
            val_working = standard_scaler.transform(val_working)
        test_working = standard_scaler.transform(test_working)

    if feature_reduction == "pca":
        pca = PCA(n_components=d_in, random_state=seed)
        train_working = pca.fit_transform(train_working)
        if val_working is not None:
            val_working = pca.transform(val_working)
        test_working = pca.transform(test_working)

    if scale in {"minmax", "standard_then_minmax"}:
        minmax_scaler = MinMaxScaler()
        train_working = minmax_scaler.fit_transform(train_working)
        if val_working is not None:
            val_working = minmax_scaler.transform(val_working)
        test_working = minmax_scaler.transform(test_working)

    return (
        train_working.astype(np.float32),
        None if val_working is None else val_working.astype(np.float32),
        test_working.astype(np.float32),
    )


def _make_tensor_dataset(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )

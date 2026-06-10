"""
Repository structure object for the adversarial-homr project.

This file provides a literal Python object `REPO_STRUCTURE` that mirrors the
repository tree (folders and files). It lists every `.py` and `.ipynb` file
explicitly and includes all folders. Large data folders are noted in
`OMITTED_PATHS` and are not enumerated file-by-file.
"""

REPO_NAME = "adversarial-homr"

OMITTED_PATHS = [
    # Large/generated dataset areas: keep the folder node, but do not enumerate
    # individual files or nested folders.
    "dataset",
    "dataset/images",
    "dataset/mxl",
    "dataset/reference_mxl",
    "dataset/cached_crops",
    "dataset/cached_prepared_staffs",

    # Batch-generated augmented page images can be huge. The wildcard means:
    # distillation/batches/<any batch name>/augmented_pages
    "distillation/batches/*/augmented_pages",
    "distillation/batches/*/rendered_images",
    "distillation/batches/*/prepared_staffs",

    "results/figures",
    "results/logs",
    "results/plots",
    "results/tables",
    "models/onnx",
    "training/onnx",
]


REPO_STRUCTURE = {
    "type": "dir",
    "name": REPO_NAME,
    "path": ".",
    "meta": {"description": "Root of the adversarial-homr project: adversarial robustness experiments and tools for HOMR"},
    "children": [
        {"type": "file", "name": "Changelog.md", "path": "Changelog.md", "meta": {"description": "Project changelog."}},
        {"type": "file", "name": "README.md", "path": "README.md", "meta": {"description": "Project overview, installation and usage."}},
        {"type": "file", "name": "pyproject.toml", "path": "pyproject.toml", "meta": {"description": "Python project config and dependencies."}},
        {"type": "file", "name": "mise.toml", "path": "mise.toml", "meta": {"description": "Environment / build metadata file."}},
        {"type": "file", "name": "Dockerfile", "path": "Dockerfile", "meta": {"description": "Dockerfile for containerizing the project."}},
        {"type": "file", "name": "Dockerfile.gpu", "path": "Dockerfile.gpu", "meta": {"description": "GPU-enabled Dockerfile."}},
        {"type": "file", "name": "Makefile", "path": "Makefile", "meta": {"description": "Convenience make targets for common tasks."}},
        {"type": "file", "name": "colab.ipynb", "path": "colab.ipynb", "meta": {"description": "Colab notebook with example runs."}},
        {"type": "file", "name": "check_onnx.ipynb", "path": "check_onnx.ipynb", "meta": {"description": "Notebook to inspect or validate ONNX models."}},

        {
            "type": "dir",
            "name": "adversarial plan",
            "path": "adversarial plan",
            "meta": {"description": "Human-readable plan and protocol files for adversarial benchmark (markdown)."},
            "children": [
                {"type": "file", "name": "homr_adversarial_robustness_protocol.md", "path": "adversarial plan/homr_adversarial_robustness_protocol.md", "meta": {"description": "Protocol for adversarial robustness experiments."}},
            ],
        },

        {
            "type": "dir",
            "name": "attacks",
            "path": "attacks",
            "meta": {"description": "Adversarial attack implementations, sweep scripts and configs."},
            "children": [
                {"type": "file", "name": "__init__.py", "path": "attacks/__init__.py", "meta": {"description": "Package initializer for attacks module."}},
                {"type": "file", "name": "HOMR_Adversarial_Benchmark_Blueprint.md", "path": "attacks/HOMR_Adversarial_Benchmark_Blueprint.md", "meta": {"description": "Blueprint/notes for the benchmark."}},
                {"type": "file", "name": "run_spectral_sweep.py", "path": "attacks/run_spectral_sweep.py", "meta": {"description": "Script to run spectral noise attack sweeps."}},
                {"type": "file", "name": "run_square_sweep.py", "path": "attacks/run_square_sweep.py", "meta": {"description": "Script to run square (patch) attack sweeps."}},
                {"type": "file", "name": "spectral_noise_injection.ipynb", "path": "attacks/spectral_noise_injection.ipynb", "meta": {"description": "Notebook demonstrating spectral noise injection attack."}},
                {
                    "type": "dir",
                    "name": "config",
                    "path": "attacks/config",
                    "meta": {"description": "Configuration files for attack sweeps (YAML)."},
                    "children": [
                        {"type": "file", "name": "sweep_parameters.yaml", "path": "attacks/config/sweep_parameters.yaml", "meta": {"description": "Parameters used for sweep runs."}},
                    ],
                },
                {
                    "type": "dir",
                    "name": "src",
                    "path": "attacks/src",
                    "meta": {"description": "Source code for attack implementations and helpers."},
                    "children": [
                        {"type": "file", "name": "__init__.py", "path": "attacks/src/__init__.py", "meta": {"description": "Package init for attacks.src."}},
                        {"type": "file", "name": "segmentation_onnx.py", "path": "attacks/src/segmentation_onnx.py", "meta": {"description": "ONNX-based segmentation utilities used by attacks."}},
                        {"type": "file", "name": "homr_wrapper.py", "path": "attacks/src/homr_wrapper.py", "meta": {"description": "Wrapper to call HOMR inference from attack scripts."}},
                        {"type": "file", "name": "square_attack.py", "path": "attacks/src/square_attack.py", "meta": {"description": "Implementation of square / patch attack."}},
                        {"type": "file", "name": "statistics_engine.py", "path": "attacks/src/statistics_engine.py", "meta": {"description": "Collects and computes statistics from attack runs."}},
                    ],
                },
            ],
        },

        {
            "type": "dir",
            "name": "dataset",
            "path": "dataset",
            "meta": {
                "description": "Dataset directory. Individual files and nested folders are intentionally omitted because this area can contain large/raw/generated data.",
                "omit_listing": True,
            },
            "children": [],
        },

        {
            "type": "dir",
            "name": "docs",
            "path": "docs",
            "meta": {"description": "Documentation source for Sphinx; conf and docs pages."},
            "children": [
                {"type": "file", "name": "conf.py", "path": "docs/conf.py", "meta": {"description": "Sphinx configuration."}},
                {"type": "file", "name": "index.rst", "path": "docs/index.rst", "meta": {"description": "Sphinx index page."}},
            ],
        },

        {"type": "dir", "name": "evaluation", "path": "evaluation", "meta": {"description": "Evaluation reports and utilities (markdown placeholders)."}, "children": [{"type": "file", "name": "empty.md", "path": "evaluation/empty.md", "meta": {"description": "Placeholder file for evaluation notes."}}]},

        {"type": "dir", "name": "experiments", "path": "experiments", "meta": {"description": "Experiment notes and artifacts (placeholder)."}, "children": [{"type": "file", "name": "empty.md", "path": "experiments/empty.md", "meta": {"description": "Placeholder file for experiments."}}]},

        {
            "type": "dir",
            "name": "homr",
            "path": "homr",
            "meta": {"description": "Core HOMR codebase: image processing, staff detection, parsing and model glue."},
            "children": [
                {"type": "file", "name": "__init__.py", "path": "homr/__init__.py", "meta": {"description": "Package initializer for homr utilities."}},
                {"type": "file", "name": "autocrop.py", "path": "homr/autocrop.py", "meta": {"description": "Automatic cropping utilities for staff regions."}},
                {"type": "file", "name": "bar_line_detection.py", "path": "homr/bar_line_detection.py", "meta": {"description": "Detects bar lines in staff images."}},
                {"type": "file", "name": "bounding_boxes.py", "path": "homr/bounding_boxes.py", "meta": {"description": "Bounding box utilities for detected symbols."}},
                {"type": "file", "name": "brace_dot_detection.py", "path": "homr/brace_dot_detection.py", "meta": {"description": "Detects brace/dot markers in staves."}},
                {"type": "file", "name": "circle_of_fifths.py", "path": "homr/circle_of_fifths.py", "meta": {"description": "Utility for key signature handling / circle of fifths."}},
                {"type": "file", "name": "color_adjust.py", "path": "homr/color_adjust.py", "meta": {"description": "Color normalization and adjustments for images."}},
                {"type": "file", "name": "constants.py", "path": "homr/constants.py", "meta": {"description": "Project-wide constants used by HOMR."}},
                {"type": "file", "name": "debug.py", "path": "homr/debug.py", "meta": {"description": "Debug helpers and tools."}},
                {"type": "file", "name": "download_utils.py", "path": "homr/download_utils.py", "meta": {"description": "Helpers to download external assets or models."}},
                {"type": "file", "name": "find_peaks.py", "path": "homr/find_peaks.py", "meta": {"description": "Signal-processing helpers used for staff/peak detection."}},
                {"type": "file", "name": "image_utils.py", "path": "homr/image_utils.py", "meta": {"description": "Image I/O and preprocessing utilities."}},
                {"type": "file", "name": "main.py", "path": "homr/main.py", "meta": {"description": "CLI / main entrypoints for HOMR tools."}},
                {"type": "file", "name": "model.py", "path": "homr/model.py", "meta": {"description": "Model wrapper and inference helpers."}},
                {"type": "file", "name": "music_xml_generator.py", "path": "homr/music_xml_generator.py", "meta": {"description": "Generate MusicXML output from model predictions."}},
                {"type": "file", "name": "noise_filtering.py", "path": "homr/noise_filtering.py", "meta": {"description": "Noise filtering utilities for images."}},
                {"type": "file", "name": "note_detection.py", "path": "homr/note_detection.py", "meta": {"description": "Detects note symbols from segmentation outputs."}},
                {"type": "file", "name": "resize.py", "path": "homr/resize.py", "meta": {"description": "Image resizing helpers."}},
                {"type": "file", "name": "simple_logging.py", "path": "homr/simple_logging.py", "meta": {"description": "Thin logging helpers used across HOMR."}},
                {"type": "file", "name": "staff_detection.py", "path": "homr/staff_detection.py", "meta": {"description": "Detect staff lines and regions in images."}},
                {"type": "file", "name": "staff_dewarping.py", "path": "homr/staff_dewarping.py", "meta": {"description": "Dewarping logic for skewed staff images."}},
                {"type": "file", "name": "staff_parsing.py", "path": "homr/staff_parsing.py", "meta": {"description": "Parse staff regions into higher-level structures."}},
                {"type": "file", "name": "staff_parsing_tromr.py", "path": "homr/staff_parsing_tromr.py", "meta": {"description": "Tromr-specific staff parsing utilities."}},
                {"type": "file", "name": "staff_position_save_load.py", "path": "homr/staff_position_save_load.py", "meta": {"description": "Save/load helpers for detected staff positions."}},
                {"type": "file", "name": "staff_regions.py", "path": "homr/staff_regions.py", "meta": {"description": "Helpers to manage detected staff region boundaries."}},
                {"type": "file", "name": "title_detection.py", "path": "homr/title_detection.py", "meta": {"description": "Detect title region on page images."}},
                {"type": "file", "name": "type_definitions.py", "path": "homr/type_definitions.py", "meta": {"description": "Shared type definitions and small dataclasses."}},
                {
                    "type": "dir",
                    "name": "segmentation",
                    "path": "homr/segmentation",
                    "meta": {"description": "Segmentation network utilities and inference code."},
                    "children": [
                        {"type": "file", "name": "__init__.py", "path": "homr/segmentation/__init__.py", "meta": {"description": "Segmentation package init."}},
                        {"type": "file", "name": "inference_segnet.py", "path": "homr/segmentation/inference_segnet.py", "meta": {"description": "Run segmentation inference using segnet models."}},
                        {"type": "file", "name": "config.py", "path": "homr/segmentation/config.py", "meta": {"description": "Segmentation model configuration."}},
                    ],
                },
                {
                    "type": "dir",
                    "name": "transformer",
                    "path": "homr/transformer",
                    "meta": {"description": "Transformer model support code and utilities."},
                    "children": [
                        {"type": "file", "name": "__init__.py", "path": "homr/transformer/__init__.py", "meta": {"description": "Transformer package init."}},
                        {"type": "file", "name": "vocabulary.py", "path": "homr/transformer/vocabulary.py", "meta": {"description": "Vocabulary construction and token handling."}},
                        {"type": "file", "name": "utils.py", "path": "homr/transformer/utils.py", "meta": {"description": "Transformer helper utilities."}},
                        {"type": "file", "name": "staff2score.py", "path": "homr/transformer/staff2score.py", "meta": {"description": "Conversion helpers from staff representation to score tokens."}},
                        {"type": "file", "name": "encoder_inference.py", "path": "homr/transformer/encoder_inference.py", "meta": {"description": "Encoder inference utilities."}},
                        {"type": "file", "name": "decoder_inference.py", "path": "homr/transformer/decoder_inference.py", "meta": {"description": "Decoder inference utilities."}},
                        {"type": "file", "name": "configs.py", "path": "homr/transformer/configs.py", "meta": {"description": "Transformer model configurations."}},
                    ],
                },
            ],
        },

        {"type": "dir", "name": "models", "path": "models", "meta": {"description": "Models, model artifacts and ONNX files."}, "children": [{"type": "dir", "name": "onnx", "path": "models/onnx", "meta": {"description": "ONNX model files and conversions.", "omit_listing": True}, "children": []}]},

        {"type": "dir", "name": "notebooks", "path": "notebooks", "meta": {"description": "Misc project notebooks and experiments (placeholder)."}, "children": [{"type": "file", "name": "empty.md", "path": "notebooks/empty.md", "meta": {"description": "Placeholder for notebooks folder."}}]},

        {"type": "dir", "name": "report", "path": "report", "meta": {"description": "Report artifacts (placeholder)."}, "children": [{"type": "file", "name": "empty.md", "path": "report/empty.md", "meta": {"description": "Placeholder for report folder."}}]},

        {"type": "dir", "name": "results", "path": "results", "meta": {"description": "Output artifacts from experiments: figures, logs and plots."}, "children": [
            {"type": "dir", "name": "figures", "path": "results/figures", "meta": {"description": "Saved figure images.", "omit_listing": True}, "children": []},
            {"type": "dir", "name": "logs", "path": "results/logs", "meta": {"description": "Run logs.", "omit_listing": True}, "children": []},
            {"type": "dir", "name": "plots", "path": "results/plots", "meta": {"description": "Saved plot files.", "omit_listing": True}, "children": []},
            {"type": "dir", "name": "tables", "path": "results/tables", "meta": {"description": "Result tables.", "omit_listing": True}, "children": []},
        ]},

        {"type": "dir", "name": "scripts", "path": "scripts", "meta": {"description": "Utility scripts and one-off helpers."}, "children": [{"type": "file", "name": "empty.md", "path": "scripts/empty.md", "meta": {"description": "Placeholder script folder file."}}]},

        {
            "type": "dir",
            "name": "tests",
            "path": "tests",
            "meta": {"description": "Unit tests for the project."},
            "children": [
                {"type": "file", "name": "__init__.py", "path": "tests/__init__.py", "meta": {"description": "Tests package initializer."}},
                {"type": "file", "name": "test_bounding_boxes.py", "path": "tests/test_bounding_boxes.py", "meta": {"description": "Tests for bounding box utilities."}},
                {"type": "file", "name": "test_circle_of_fifths.py", "path": "tests/test_circle_of_fifths.py", "meta": {"description": "Tests for circle of fifths utilities."}},
                {"type": "file", "name": "test_humdrum_kern_parser.py", "path": "tests/test_humdrum_kern_parser.py", "meta": {"description": "Tests for humdrum/kern parsing."}},
                {"type": "file", "name": "test_mix_datasets.py", "path": "tests/test_mix_datasets.py", "meta": {"description": "Tests for dataset mixing utilities."}},
                {"type": "file", "name": "test_model.py", "path": "tests/test_model.py", "meta": {"description": "Model-related unit tests."}},
                {"type": "file", "name": "test_music_xml_generator.py", "path": "tests/test_music_xml_generator.py", "meta": {"description": "Tests for MusicXML generation."}},
                {"type": "file", "name": "test_music_xml_parser.py", "path": "tests/test_music_xml_parser.py", "meta": {"description": "Tests for MusicXML parsing utilities."}},
                {"type": "file", "name": "test_poetry_config.py", "path": "tests/test_poetry_config.py", "meta": {"description": "Configuration tests for poetry/packaging."}},
                {"type": "file", "name": "test_primus_semantic_parser.py", "path": "tests/test_primus_semantic_parser.py", "meta": {"description": "Tests for Primus semantic parser."}},
                {"type": "file", "name": "test_staff_detection.py", "path": "tests/test_staff_detection.py", "meta": {"description": "Staff detection tests."}},
                {"type": "file", "name": "test_staff_merging.py", "path": "tests/test_staff_merging.py", "meta": {"description": "Tests for staff merging routines."}},
                {"type": "file", "name": "test_staff_regions.py", "path": "tests/test_staff_regions.py", "meta": {"description": "Tests for staff region helpers."}},
                {"type": "file", "name": "test_title_detection.py", "path": "tests/test_title_detection.py", "meta": {"description": "Title detection tests."}},
                {"type": "file", "name": "test_training_vocabulary.py", "path": "tests/test_training_vocabulary.py", "meta": {"description": "Vocabulary training tests."}},
                {"type": "file", "name": "test_vocabulary.py", "path": "tests/test_vocabulary.py", "meta": {"description": "Vocabulary unit tests."}},
            ],
        },

        {
            "type": "dir",
            "name": "training",
            "path": "training",
            "meta": {"description": "Training scripts, dataset conversions and ONNX packaging."},
            "children": [
                {"type": "file", "name": "__init__.py", "path": "training/__init__.py", "meta": {"description": "Training package init."}},
                {"type": "file", "name": "download.py", "path": "training/download.py", "meta": {"description": "Download and prepare pretrained resources."}},
                {"type": "file", "name": "run_id.py", "path": "training/run_id.py", "meta": {"description": "Utilities for run identifiers and metadata."}},
                {"type": "file", "name": "show_examples_from_index.py", "path": "training/show_examples_from_index.py", "meta": {"description": "Display dataset examples by index."}},
                {"type": "file", "name": "train.py", "path": "training/train.py", "meta": {"description": "High-level training entrypoint."}},
                {
                    "type": "dir",
                    "name": "architecture",
                    "path": "training/architecture",
                    "meta": {"description": "Model architecture code used for training and experiments."},
                    "children": [
                        {
                            "type": "dir",
                            "name": "segmentation",
                            "path": "training/architecture/segmentation",
                            "meta": {"description": "Segmentation model architectures."},
                            "children": [{"type": "file", "name": "model.py", "path": "training/architecture/segmentation/model.py", "meta": {"description": "Segmentation model definition."}}],
                        },
                        {
                            "type": "dir",
                            "name": "transformer",
                            "path": "training/architecture/transformer",
                            "meta": {"description": "Transformer architectures used for sequence prediction."},
                            "children": [
                                {"type": "file", "name": "__init__.py", "path": "training/architecture/transformer/__init__.py", "meta": {"description": "Transformer architecture package init."}},
                                {"type": "file", "name": "tromr_arch.py", "path": "training/architecture/transformer/tromr_arch.py", "meta": {"description": "Tromr-specific transformer architecture."}},
                                {"type": "file", "name": "staff2score.py", "path": "training/architecture/transformer/staff2score.py", "meta": {"description": "Staff-to-score conversion used in architectures."}},
                                {"type": "file", "name": "encoder.py", "path": "training/architecture/transformer/encoder.py", "meta": {"description": "Encoder implementation."}},
                                {"type": "file", "name": "decoder.py", "path": "training/architecture/transformer/decoder.py", "meta": {"description": "Decoder implementation."}},
                                {"type": "file", "name": "custom_x_transformer.py", "path": "training/architecture/transformer/custom_x_transformer.py", "meta": {"description": "Custom transformer classes."}},
                            ],
                        },
                    ],
                },
                {
                    "type": "dir",
                    "name": "datasets",
                    "path": "training/datasets",
                    "meta": {"description": "Dataset conversion and parsing utilities used for building training sets."},
                    "children": [
                        {"type": "file", "name": "convert_primus.py", "path": "training/datasets/convert_primus.py", "meta": {"description": "Convert Primus dataset format."}},
                        {"type": "file", "name": "convert_lieder.py", "path": "training/datasets/convert_lieder.py", "meta": {"description": "Convert Lieder dataset format."}},
                        {"type": "file", "name": "convert_grandstaff.py", "path": "training/datasets/convert_grandstaff.py", "meta": {"description": "Convert grandstaff dataset format."}},
                        {"type": "file", "name": "music_xml_parser.py", "path": "training/datasets/music_xml_parser.py", "meta": {"description": "Parser for MusicXML files into training format."}},
                        {"type": "file", "name": "musescore_svg.py", "path": "training/datasets/musescore_svg.py", "meta": {"description": "Helpers to parse MuseScore SVG outputs."}},
                        {"type": "file", "name": "humdrum_kern_parser.py", "path": "training/datasets/humdrum_kern_parser.py", "meta": {"description": "Parser for humdrum/kern dataset files."}},
                        {"type": "file", "name": "staff_merging.py", "path": "training/datasets/staff_merging.py", "meta": {"description": "Merge staff fragments into consistent examples."}},
                        {"type": "file", "name": "primus_semantic_parser.py", "path": "training/datasets/primus_semantic_parser.py", "meta": {"description": "Semantic parser for Primus dataset."}},
                        {"type": "file", "name": "unzip_datasets.py", "path": "training/datasets/unzip_datasets.py", "meta": {"description": "Helpers to unzip and prepare raw dataset archives."}},
                        {"type": "file", "name": "zip_datasets.py", "path": "training/datasets/zip_datasets.py", "meta": {"description": "Pack datasets into zip archives for distribution."}},
                        {"type": "file", "name": "__init__.py", "path": "training/datasets/__init__.py", "meta": {"description": "Datasets package init."}},
                    ],
                },
                {"type": "file", "name": "validate_music_xml_conversion.py", "path": "training/validate_music_xml_conversion.py", "meta": {"description": "Validate MusicXML conversion correctness."}},
                {
                    "type": "dir",
                    "name": "transformer",
                    "path": "training/transformer",
                    "meta": {"description": "Transformer training scripts and helpers."},
                    "children": [
                        {"type": "file", "name": "__init__.py", "path": "training/transformer/__init__.py", "meta": {"description": "Transformer training package init."}},
                        {"type": "file", "name": "training_vocabulary.py", "path": "training/transformer/training_vocabulary.py", "meta": {"description": "Build training vocabulary for transformer."}},
                        {"type": "file", "name": "train.py", "path": "training/transformer/train.py", "meta": {"description": "Transformer training script."}},
                        {"type": "file", "name": "mix_datasets.py", "path": "training/transformer/mix_datasets.py", "meta": {"description": "Utilities to mix datasets for transformer training."}},
                        {"type": "file", "name": "metrics.py", "path": "training/transformer/metrics.py", "meta": {"description": "Training metrics to evaluate model performance."}},
                        {"type": "file", "name": "image_utils.py", "path": "training/transformer/image_utils.py", "meta": {"description": "Image utilities used specifically by transformer training."}},
                        {"type": "file", "name": "data_loader.py", "path": "training/transformer/data_loader.py", "meta": {"description": "Data loader for transformer training."}},
                    ],
                },
                {
                    "type": "dir",
                    "name": "segmentation",
                    "path": "training/segmentation",
                    "meta": {"description": "Segmentation-specific training code and dataset definitions."},
                    "children": [
                        {"type": "file", "name": "__init__.py", "path": "training/segmentation/__init__.py", "meta": {"description": "Segmentation training package init."}},
                        {"type": "file", "name": "train.py", "path": "training/segmentation/train.py", "meta": {"description": "Segmentation training entrypoint."}},
                        {"type": "file", "name": "dense_dataset_definitions.py", "path": "training/segmentation/dense_dataset_definitions.py", "meta": {"description": "Dataset definitions for dense segmentation training."}},
                        {"type": "file", "name": "build_label.py", "path": "training/segmentation/build_label.py", "meta": {"description": "Build segmentation labels from annotations."}},
                    ],
                },
                {
                    "type": "dir",
                    "name": "onnx",
                    "path": "training/onnx",
                    "meta": {"description": "ONNX model packaging and conversion utilities (model files omitted)."},
                    "children": [
                        {"type": "file", "name": "__init__.py", "path": "training/onnx/__init__.py", "meta": {"description": "ONNX packaging package init."}},
                        {"type": "file", "name": "main.py", "path": "training/onnx/main.py", "meta": {"description": "Main script for ONNX export/processing."}},
                        {"type": "file", "name": "fuse.py", "path": "training/onnx/fuse.py", "meta": {"description": "Fuse and simplify model graphs."}},
                        {"type": "file", "name": "convert.py", "path": "training/onnx/convert.py", "meta": {"description": "Convert models to ONNX format."}},
                        {"type": "file", "name": "simplify.py", "path": "training/onnx/simplify.py", "meta": {"description": "Simplify ONNX graphs."}},
                        {"type": "file", "name": "quantization.py", "path": "training/onnx/quantization.py", "meta": {"description": "Quantize ONNX models for efficiency."}},
                        {"type": "file", "name": "split_weights.py", "path": "training/onnx/split_weights.py", "meta": {"description": "Split large weight files for packaging."}},
                    ],
                },
            ],
        },

        {
            "type": "dir",
            "name": "validation",
            "path": "validation",
            "meta": {"description": "Validation utilities and metrics for model outputs."},
            "children": [
                {"type": "file", "name": "__init__.py", "path": "validation/__init__.py", "meta": {"description": "Validation package init."}},
                {"type": "file", "name": "rate_validation_result.py", "path": "validation/rate_validation_result.py", "meta": {"description": "Rate and summarize validation results."}},
                {"type": "file", "name": "symbol_error_rate_torch.py", "path": "validation/symbol_error_rate_torch.py", "meta": {"description": "Compute symbol error rates using torch."}},
            ],
        },

        {"type": "dir", "name": "attacks/notebooks", "path": "attacks/notebooks", "meta": {"description": "Notebooks used for attack experiments."}, "children": [{"type": "file", "name": "02_segmentation_smoke_test.ipynb", "path": "attacks/notebooks/02_segmentation_smoke_test.ipynb", "meta": {"description": "Segmentation smoke-test notebook."}}]},

        {"type": "file", "name": ".tmp_check_imports.py", "path": ".tmp_check_imports.py", "meta": {"description": "Temporary import-check helper script."}},
    ],
}


if __name__ == "__main__":
    import json
    print(json.dumps(REPO_STRUCTURE, indent=2))

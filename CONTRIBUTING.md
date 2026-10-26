# Contributing

We're thrilled you're considering contributing to `homr`! This document outlines the process for contributing to the `homr` project.

## Getting Started

Before you begin, make sure you have a GitHub account and have familiarized yourself with the `homr` project by reading through the documentation and exploring the codebase.

### Issues

- Before submitting a new issue, please search existing issues to avoid duplicates.
- Please attach a real image: a scan or photo of actual sheet music. Synthetic or AI generated images show problems that don't occur in practice, so we can't work with them.
- Please keep it short. `homr` is a volunteer project and we read every word. A few sentences and the image tell us more than a long analysis. If an AI helps you write, trim its output to the essentials.
- Issues are for problems, pull requests are for solutions. If you have a fix, open a pull request: before, together with or without an issue is all fine.

### Pull Requests

1. **Fork the Repository**: Start by forking the `homr` repository to your GitHub account.
2. **Create a Branch**: Create a branch in your forked repository for your contribution.
3. **Make Your Changes**: Implement your changes, run `make format` to apply the code style and `make ci` to run code analysis and tests
4. **Commit Your Changes**: Make sure your commit messages are clear and follow best practices.
5. **Push Your Changes**: Push your changes to your forked repository on GitHub.
6. **Submit a Pull Request**: Open a pull request from your forked repository to the main `homr` repository. Provide a clear description of the problem you're solving or the feature you're adding.

## Contribution Guidelines

### Code Style

- Follow the coding style and conventions already present in the `homr` codebase.
- Use meaningful variable names and comments to make the code as readable as possible.
- Format your code by running `make format`.

### Testing

- Add unit tests for new features or bug fixes whenever possible.
- Ensure that all tests pass before submitting your pull request, run `make ci`.

### Documentation

- Update the README.md or documentation if your changes require it.
- Add comments to your code where necessary to explain complex or non-obvious parts of your implementation.

### AI usage
Please explain how you used AI. For example to what extent you verified the AI's output.

### Detection Algorithm Changes

If your contribution impacts the detection algorithm, please include examples that showcase the enhancements. 
This should cover both qualitative examples and, where possible, quantitative metrics to illustrate the improvements. 
Utilize the [rate_validation_result.py](https://github.com/liebharc/homr/blob/main/validation/rate_validation_result.py) script or our benchmark (for more details see [Benchmark.md](https://github.com/liebharc/homr/blob/main/Benchmark.md)) to generate metrics across multiple images, 
provided you have access to corresponding groundtruth MusicXML files.

### Training Model Updates

If your contribution involves modifications to the model training process, please include a link to the updated trained model. 
This ensures that reviewers and users can easily access and evaluate the impact of your changes.

## Licensing

By contributing to `homr`, you agree that your contributions will be licensed under both the AGPL and Apache 2 licenses. This ensures that your contributions can be used in a wide range of applications while maintaining the project's open-source nature.

## Questions?

If you have any questions or need further clarification on the contribution process, feel free to open an issue for discussion.

Thank you for contributing to `homr`!

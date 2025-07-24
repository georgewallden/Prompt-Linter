# PromptLinter

A PyTorch-powered linter for LLM prompts that analyzes prompt quality, scores hallucination risk, provides an explanatory visual trace, and suggests improvements.

## Overview

This project is a comprehensive system designed to provide a "pre-flight check" for Large Language Model (LLM) prompts. Before sending a prompt to an expensive or powerful LLM, this tool analyzes its quality and safety, providing actionable feedback to the user. This helps mitigate frustrating trial-and-error, reduces the risk of model hallucination, and guides users toward writing more effective prompts.

## Project Structure

The repository is organized to separate the core logic from the applications that use it (training, serving) and the infrastructure code.

-   **/src/**: Contains all primary Python source code for the application.
    -   **/src/backend/engine/**: The core, reusable analysis engine. This is the heart of the project, completely decoupled from any web framework or training script.
    -   **/src/backend/api/**: The FastAPI web server that wraps the `engine` and exposes it over an HTTP API.
    -   **/src/frontend/**: A designated place for any future UI components.
-   **/training/**: Contains all scripts related to data processing, model training, and evaluation (Phase 1).
-   **/data/**: A temporary local directory for raw datasets before they are uploaded to cloud storage.
-   **/terraform/**: Contains all Infrastructure as Code (IaC) files for managing AWS resources.
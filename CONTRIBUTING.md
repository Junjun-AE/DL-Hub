# Contributing to DL-Hub

Thank you for your interest in contributing! Here's how you can help.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests if applicable
5. Commit: `git commit -m "feat: add my feature"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

## Development Setup

```bash
git clone https://github.com/YOUR_ORG/Deep_learning_tools4.git
cd Deep_learning_tools4
pip install -r dlhub_project/requirements.txt

# Frontend development
cd dlhub_project/dlhub/frontend
npm install
npm run dev    # Dev server on :3000
```

## Project Structure

Each task module follows a consistent structure:

```
model_image_<task>/
├── app.py              # Gradio web app (entry point)
├── config/             # Model registry and configuration
├── models/             # Model factory
├── engine/             # Training engine with callbacks
├── data/               # Dataset and data conversion
├── utils/              # Environment/data validators
└── README.md           # Module documentation
```

## Code Style

- Python: Follow PEP 8; use type hints for function signatures
- JavaScript/React: Use functional components with hooks
- Comments: Chinese is acceptable for domain-specific comments; docstrings should be bilingual when possible

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Build/tooling changes

## Adding a New Task

1. Create `model_image_<newtask>/` following the standard structure
2. Implement `app.py` with DL-Hub adapter integration (see `dlhub_project/dlhub/app_adapters/INTEGRATION_GUIDE.md`)
3. Add task type to `TaskService.VALID_TASK_TYPES` in `dlhub_project/dlhub/backend/services/task_service.py`
4. Add a card to `SOFTWARE_CARDS` in `dlhub_project/dlhub/frontend/src/App.jsx`
5. Create user manual HTML in `dlhub_project/user_manual/`

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0 (or AGPL-3.0 for changes to `model_image_sevseg/ultralytics/`).

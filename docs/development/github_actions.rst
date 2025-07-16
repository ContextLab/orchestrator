GitHub Actions Setup
====================

The Orchestrator project uses GitHub Actions for continuous integration and automated badge generation.

Setting Up Coverage Badge
-------------------------

To enable the coverage badge in your README, follow these steps:

1. **Create a GitHub Gist**:
   
   - Go to https://gist.github.com
   - Create a new secret gist with any content (it will be overwritten)
   - Copy the gist ID from the URL

2. **Create a Personal Access Token**:
   
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create a new token with ``gist`` scope
   - Copy the token

3. **Add Secret to Repository**:
   
   - Go to your repository settings → Secrets and variables → Actions
   - Add a new secret named ``GIST_SECRET`` with your personal access token

4. **Update Workflow File**:
   
   - Edit ``.github/workflows/coverage.yml``
   - Replace ``<YOUR_GIST_ID>`` with your actual gist ID

5. **Update README Badge**:
   
   - The coverage badge URL should point to:
     ``https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/YOUR_USERNAME/YOUR_GIST_ID/raw/orchestrator-coverage.json``

Workflow Files
--------------

Tests Workflow
~~~~~~~~~~~~~~

The tests workflow (``.github/workflows/tests.yml``) runs on every push and pull request:

- Tests on multiple Python versions (3.8-3.12)
- Tests on multiple operating systems (Ubuntu, macOS, Windows)
- Runs linting and formatting checks
- Ensures code quality across platforms

Coverage Workflow
~~~~~~~~~~~~~~~~~

The coverage workflow (``.github/workflows/coverage.yml``) runs on main branch:

- Calculates test coverage percentage
- Updates the coverage badge in README
- Optionally uploads to Codecov for detailed reports

Manual Badge Updates
--------------------

If you prefer to update badges manually, you can extract coverage from the test output:

.. code-block:: bash

   # Run tests with coverage
   pytest --cov=src/orchestrator --cov-report=term
   
   # The output will show coverage percentage
   # Update the README badge URL with the percentage

Available Badges
----------------

The project uses several badges:

- **PyPI Version**: Shows latest published version
- **Python Versions**: Shows supported Python versions
- **Downloads**: Shows monthly download count
- **License**: Shows project license
- **Tests**: Shows if tests are passing
- **Coverage**: Shows test coverage percentage
- **Documentation**: Shows documentation build status

Badge Customization
-------------------

You can customize badge colors and styles:

.. code-block:: markdown

   # Different styles
   ![Badge](https://img.shields.io/badge/style-flat-green)
   ![Badge](https://img.shields.io/badge/style-flat--square-green?style=flat-square)
   ![Badge](https://img.shields.io/badge/style-for--the--badge-green?style=for-the-badge)
   
   # Custom colors
   ![Badge](https://img.shields.io/badge/custom-color-ff69b4)
   ![Badge](https://img.shields.io/badge/custom-color-blueviolet)
PACKAGE_CODE = app_predictor.py

format:
	autoflake --recursive --in-place --remove-all-unused-imports $(PACKAGE_CODE)
	isort $(PACKAGE_CODE)
	black --line-length=140 $(PACKAGE_CODE)

lint:
	flake8 --jobs 1 --statistics --show-source $(PACKAGE_CODE)
	pylint --jobs 1 --rcfile=setup.cfg $(PACKAGE_CODE)
	black --line-length=140 --check $(PACKAGE_CODE)

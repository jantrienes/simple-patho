CODE=simplepatho scripts
TESTS=tests
COCHRANE_URL=https://raw.githubusercontent.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts/edf6504ea28b2458ec6b4c172482ad15387aeeef/data/data-1024/

format:
	black ${CODE} ${TESTS}
	isort ${CODE} ${TESTS}

test:
	python -m nltk.downloader punkt
	pytest --cov-report html --cov=${CODE} ${CODE} ${TESTS}

lint:
	pylint --disable=R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}

lintci:
	pylint --disable=W,R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}

cochrane:
	mkdir -p data/raw/cochrane/
	wget -nc -P data/raw/cochrane/ \
		${COCHRANE_URL}/test.doi \
		${COCHRANE_URL}/test.source \
		${COCHRANE_URL}/test.target \
		${COCHRANE_URL}/val.doi \
		${COCHRANE_URL}/val.source \
		${COCHRANE_URL}/val.target \
		${COCHRANE_URL}/train.doi \
		${COCHRANE_URL}/train.source \
		${COCHRANE_URL}/train.target \

	python -m simplepatho.convert_data --raw_path data/raw/cochrane --out_path data/processed/cochrane

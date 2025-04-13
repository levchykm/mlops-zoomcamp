FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip && pip install pipenv && pip install pyarrow && pip install pandas

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "homework_04.py", "homework_04.py" ]

ENTRYPOINT [ "python", "homework_04.py" ]

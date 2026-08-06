FROM public.ecr.aws/lambda/python:3.12

# opencv-python-headless needs no GUI system libs (libGL etc.) — nothing to install here

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py ./

CMD ["handler.handler"]

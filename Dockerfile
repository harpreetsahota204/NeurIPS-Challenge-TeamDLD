FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

RUN apt update  && apt install -y git


RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/workspace

COPY --chown=user . $HOME/workspace

COPY ./deci-lm/requirements.txt ./
# COPY ./deci-lm/train.py ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Setup server requriements
COPY ./fast_api_requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# For API server
COPY ./main.py ./
COPY ./deci-lm/download_model.py ./
RUN python download_model.py



# CMD [ "python", "train.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
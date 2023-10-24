FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

RUN apt update  && apt install -y git


RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/workspace

COPY --chown=user . $HOME/workspace

COPY ./deci-lm/requirements.txt ./
COPY ./deci-lm/train.py ./
COPY ./deci-lm/app.log ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt


CMD [ "python", "train.py"]
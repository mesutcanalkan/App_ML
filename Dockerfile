# app/Dockerfile
FROM python:3.9

WORKDIR /APP_ML/

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    automake \
    ca-certificates \
    g++ \
    git \
    libtool \
    make \
    pkg-config \
    wget \
    libicu-dev \
    zlib1g-dev \
    libtiff5-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libgl1-mesa-dev \
    && apt-get install -y --no-install-recommends \
    asciidoc \
    docbook-xsl \
    xsltproc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install leptonica and tesseract
RUN wget http://www.leptonica.org/source/leptonica-1.82.0.tar.gz \
    && git clone --recursive https://github.com/tesseract-ocr/tesseract \
    && tar -zxvf leptonica-1.82.0.tar.gz \
    && cd ./leptonica-1.82.0 \
    && ./configure \
    && make \
    && make install \
    && cd ../tesseract \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install \
    && ldconfig \
    && make training \
    && make training-install

# Download language data
ENV TESSDATA_PREFIX="/usr/local/share/tessdata"
# If you want to use languages other than English and French, edit the following lines.
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata -P ${TESSDATA_PREFIX} \
    && wget https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata -P ${TESSDATA_PREFIX}


COPY requirements.txt .
RUN pip install --upgrade pip \
    # && pip install --default-timeout=1000 future \
    && pip install -r requirements.txt
RUN find /usr/local/lib/python3.9/site-packages/streamlit -type f \( -iname \*.py -o -iname \*.js \) -print0 | xargs -0 sed -i 's/healthz/health-check/g'
COPY . ./
EXPOSE 8080

CMD streamlit run --server.port 8080 --browser.serverAddress 0.0.0.0 --server.enableCORS False --server.enableXsrfProtection False app.py
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

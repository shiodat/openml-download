# OpenML dataset downloader

## Install

1. Install openml python client.
    ```
    $ git clone https://github.com/openml/openml-python
    $ cd openml-python
    $ python setup.py install
    ```

2. Install requirement packages.
    ```
    $ pip install -r requirements.txt
    ```

## How to use

1. Get your OpenML API key.

    Access https://www.openml.org/ and register your account.

2. Access your account page and go to API AUTHENTICATION, then you can find your api key.

3. Edit config at openml-download/api.yaml.

    - api_key: your api key
    - cache_dir: location of cache.
    - save_dir: location of dataset you are going to download.

4. Edit dataset list `datasets.csv`.

    - id: dataset id at OpenML. 
        - For example, this iris dataset id is 61. https://www.openml.org/d/61
    - name: dataset name (optional)

5. Download dataset.

    ```
    $ python download.py
    ```
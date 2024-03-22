# Helical Package
## Prerequisites

Have the `21iT009_051_full_data.csv` file in a base repo.
Then install the package `helical-package` like so

```
pip install git+https://github.com/helicalAI/helical-package.git
```

Copy the file `helical-package-test.py` into your repo. As the downloading steps are one-time only, you can comment those lines out as you seem fit.Execute like so:
```
python helical-test-package.py
```

# FastAPI - Endpoints
This is still work-in-progress but you the idea is to use FastAPI locally to show how we can use API endpoints to access our Helical package.
First, install the requirements with 
```
pip install -r requirements.txt
```
You can then run
```
python run_app.py
```
In your browser, go to `http://0.0.0.0:8000`

## Docker
You can also use docker. Note, this does not work on Apple Silicon (M1-M3). 
First build the container:
```
docker build -f Dockerfile -t helical-docker .
```
Run by exposing the port
```
docke run -it -p 8000:8000 helical-docker
```
In the container, run the `run_app.py`.
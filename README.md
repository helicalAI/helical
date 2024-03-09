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

# Django backend
This is still work-in-progress but you can already run django locally.
First, install the requirements with 
```
pip install -r requirements.txt
```
You can then go into the django folder and run these commands
```
cd django
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
In your browser, go to `http://127.0.0.1:8000/swagger/`
To use the helical package with these endpoints in django, make sure you have the helical package installed in your environment.

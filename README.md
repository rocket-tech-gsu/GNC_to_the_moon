# GNC_to_the_moon
In-house developed Guidance Navigation Control software libraries!

# Setup:
There are some dependencies in requirements.txt that needs an older version of python: `3.12.7`
- Make a virtual environment using that version of python using:
```
python3.12 -m venv env
```
- Activate that virtual environment:
for macOS users:
```
source env/bin/activate
```
for Windows users:
```
source env/scripts/activate
```
- Install the dependencies in this virtual environment now:
```
pip install -r requirements.txt
```
# Open Source Contribution Instructions:
Before you stage all the changes for the final time just before the pull request, make sure to update any new packages in the requirements.txt file using the following command:
```
pip freeze > requirements.txt
```
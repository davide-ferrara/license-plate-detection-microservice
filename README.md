# License Plate Detection - Microservices

![license-plate-image](docs/license-plate.jpg)

**Automatic Number Plate Recognition** (ANPR) is the process of reading the characters on the plate with various **optical character recognition** (OCR) methods by separating the plate region on the vehicle image obtained from automatic plate recognition.

This repository forks the [Automatic_Number_Plate_Recognition_YOLO_OCR
](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR) one by [mftnakrsu](https://github.com/mftnakrsu) to extract the license plate detection methods and create a microservices deployable as Docker containers.

## How to build

Build the image using the a bash script.

```bash
chmod +x build.sh
./build.sh
```

## How to run

Run the image as following.

```bash
chmod +x run.sh
./run.sh
```

## Run a stress test

```bash
chmod +x run.sh
./stress.sh dataset localhost 8080
```
![sample](https://github.com/davide-ferrara/license-plate-detection-microservice/assets/78814851/e8adf0a9-0772-43bd-bec0-22e08e4be25d)
